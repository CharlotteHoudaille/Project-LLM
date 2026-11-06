"""Fine-tuning LoRA de SmolLM2 sur les comptines.

    uv run --group lora python lora_experiments/train_lora.py --experiment style
    uv run --group lora python lora_experiments/train_lora.py --experiment keywords
    uv run --group lora python lora_experiments/train_lora.py --experiment keywords --smoke   # test rapide

Ce que fait ce script :

1. charge SmolLM2 pré-entraîné et **gèle tous ses poids** ;
2. vérifie que les couches visées par config.yaml existent bien dans le modèle ;
3. ajoute un adaptateur LoRA à chacune : à côté d'une couche figée W (d × k), deux petites
   matrices entraînables A (r × k) et B (d × r). La couche calcule alors

       h = W·x + (alpha / r) · B·A·x

   B est initialisée à zéro : au départ, le modèle adapté est exactement le modèle d'origine ;
4. entraîne seulement A et B (≈ 2 % des paramètres), avec AdamW, un échauffement puis une
   décroissance en cosinus du learning rate, et une mesure de la validation à intervalles réguliers ;
5. garde l'adaptateur du meilleur pas de validation et l'enregistre **séparément** du modèle
   de base (quelques Mo au lieu de 270), avec ses mesures et la courbe des losses.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib
import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def pick_device() -> tuple[str, dict]:
    """GPU NVIDIA si disponible (calcul en bfloat16 ou float16), sinon CPU (float32).
    On ne suppose jamais qu'un GPU existe."""
    if torch.cuda.is_available():
        bf16 = torch.cuda.is_bf16_supported()
        return f"cuda ({torch.cuda.get_device_name(0)})", {"bf16": bf16, "fp16": not bf16}
    if torch.backends.mps.is_available():
        return "mps (Apple)", {}
    return "cpu", {}


def check_target_modules(model: nn.Module, wanted: list[str]) -> list[str]:
    """Les couches linéaires du modèle dont le nom finit par l'un des noms voulus.
    Échoue clairement si un nom de config.yaml n'existe pas (autre architecture, faute de frappe)."""
    linear_names = {name.split(".")[-1] for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    missing = [w for w in wanted if w not in linear_names]
    if missing:
        raise SystemExit(f"couches absentes du modèle : {missing}. Couches linéaires disponibles : "
                         f"{sorted(linear_names)}")
    counts = {w: sum(name.endswith("." + w) for name, _ in model.named_modules()) for w in wanted}
    print("couches ciblées par LoRA :", counts, "(lm_head exclue : elle n'est pas dans les blocs)")
    return wanted


class Collator:
    """Met les exemples d'un lot à la même longueur : input_ids complétés par <|endoftext|>
    (non vus grâce à l'attention_mask), labels complétés par -100 (ignorés par la loss)."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, rows: list[dict]) -> dict:
        length = max(len(r["input_ids"]) for r in rows)
        pad = lambda seq, value: seq + [value] * (length - len(seq))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad(r["input_ids"], self.pad_id) for r in rows]),
            "attention_mask": torch.tensor([pad([1] * len(r["input_ids"]), 0) for r in rows]),
            "labels": torch.tensor([pad(r["labels"], -100) for r in rows]),
        }


def plot_losses(history: list[dict], path: Path, title: str) -> None:
    train = [(h["step"], h["loss"]) for h in history if "loss" in h]
    val = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    plt.figure(figsize=(7, 3.6))
    if train:
        plt.plot(*zip(*train), label="entraînement")
    if val:
        plt.plot(*zip(*val), marker="o", label="validation")
    plt.xlabel("pas")
    plt.ylabel("loss (par token)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA de SmolLM2 sur les comptines.")
    parser.add_argument("--experiment", choices=["style", "keywords"], required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="test rapide : 64 exemples, 10 pas, pour vérifier que tout s'enchaîne")
    # Variantes sans toucher à config.yaml (ex. : un modèle plus grand).
    parser.add_argument("--model", help="autre modèle de base (ex. HuggingFaceTB/SmolLM2-360M)")
    parser.add_argument("--epochs", type=float, help="autre nombre d'epochs")
    parser.add_argument("--batch-size", type=int, help="autre taille de lot (par pas de calcul)")
    parser.add_argument("--grad-accum", type=int, help="autre accumulation de gradient")
    parser.add_argument("--tag", default="", help="suffixe du dossier de sortie (ex. _360m)")
    args = parser.parse_args()

    seed, lora, train_cfg = CONFIG["seed"], CONFIG["lora"], dict(CONFIG["training"])
    if args.model:
        CONFIG["model"]["name"] = args.model
    for key, value in (("epochs", args.epochs), ("batch_size", args.batch_size),
                       ("gradient_accumulation", args.grad_accum)):
        if value is not None:
            train_cfg[key] = value
    set_seed(seed)
    name = args.experiment + args.tag + ("_smoke" if args.smoke else "")
    out_dir = HERE / "outputs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    device, precision = pick_device()
    print(f"expérience : {name}   appareil : {device}")

    # 1. Le modèle pré-entraîné et son tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["name"])
    model = AutoModelForCausalLM.from_pretrained(CONFIG["model"]["name"], dtype=torch.float32)
    total_params = sum(p.numel() for p in model.parameters())

    # 2-3. Vérification des couches, puis LoRA. get_peft_model gèle tous les poids d'origine.
    targets = check_target_modules(model, lora["target_modules"])
    peft_config = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                             target_modules=targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"paramètres entraînables : {trainable:,} sur {total_params:,} ({trainable / total_params:.2%})")

    # 4. Les données (prepare_data.py) et l'entraînement.
    data = load_from_disk(str(ROOT / CONFIG["data"]["out_dir"] / f"tokenized_{args.experiment}"))
    data = data.remove_columns("id")
    train_set, val_set = data["train"].shuffle(seed=seed), data["validation"]
    if args.smoke:
        train_set, val_set = train_set.select(range(64)), val_set.select(range(32))

    # Courbes dans TensorBoard, à côté de celles des notebooks : uv run tensorboard --logdir runs
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(ROOT / "runs" / f"lora_{name}")
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["scheduler"],
        warmup_steps=train_cfg["warmup_ratio"],  # Transformers 5 : une valeur < 1 est une fraction des pas
        weight_decay=train_cfg["weight_decay"],
        num_train_epochs=train_cfg["epochs"],
        max_steps=10 if args.smoke else -1,
        eval_strategy="steps",
        eval_steps=5 if args.smoke else train_cfg["eval_every"],
        logging_steps=1 if args.smoke else 10,
        save_strategy="steps",
        save_steps=5 if args.smoke else train_cfg["eval_every"],
        save_total_limit=1,
        load_best_model_at_end=True,       # on garde l'adaptateur du meilleur pas de validation
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=seed,
        dataloader_num_workers=0,
        **precision,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=val_set,
                      data_collator=Collator(tokenizer.eos_token_id))
    start = time.perf_counter()
    trainer.train()
    duration = time.perf_counter() - start

    # 5. Mesures finales (meilleur adaptateur) et sauvegarde séparée de l'adaptateur.
    val_loss = trainer.evaluate()["eval_loss"]
    test_set = data["test"].select(range(32)) if args.smoke else data["test"]
    test_loss = trainer.evaluate(test_set, metric_key_prefix="test")["test_loss"]
    model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    history = trainer.state.log_history
    plot_losses(history, out_dir / "loss.png", f"LoRA — {name}")
    metrics = {
        "experiment": name, "model": CONFIG["model"]["name"], "device": device,
        "trainable_params": trainable, "total_params": total_params, "train_seconds": round(duration, 1),
        "steps": trainer.state.global_step, "best_step": trainer.state.best_global_step,
        "val_loss": val_loss, "val_perplexity": math.exp(val_loss),
        "test_loss": test_loss, "test_perplexity": math.exp(test_loss),
        "lora": lora, "training": train_cfg, "log_history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n{name} : {trainer.state.global_step} pas en {duration:.0f} s   meilleur pas {metrics['best_step']}   "
          f"val {val_loss:.3f} (perplexité {math.exp(val_loss):.1f})   test {test_loss:.3f} "
          f"(perplexité {math.exp(test_loss):.1f})")
    print(f"adaptateur → {(out_dir / 'adapter').relative_to(ROOT)}   courbe → {(out_dir / 'loss.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
