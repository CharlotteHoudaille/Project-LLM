"""Écrit des comptines avec SmolLM2, avec ou sans adaptateur LoRA.

    uv run --group lora python lora_experiments/generate_lora.py --model keywords cat moon
    uv run --group lora python lora_experiments/generate_lora.py --model base cat moon        # sans adaptation
    uv run --group lora python lora_experiments/generate_lora.py --model style --n 3          # expérience 1

Modèles :
- base     : SmolLM2 pré-entraîné, sans adaptation, avec la consigne de l'expérience 2 ;
- style    : SmolLM2 + adaptateur de l'expérience 1 (consigne "Nursery rhyme:", sans mots-clés) ;
- keywords : SmolLM2 + adaptateur de l'expérience 2 (consigne avec les mots-clés).

L'adaptateur ne remplace rien : PeftModel le pose sur le modèle de base, qui reste intact
sur le disque. Retirer l'adaptateur rend exactement le modèle d'origine.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

HERE = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
N_LINES = 4
# Mêmes réglages de tirage que le modèle historique : température 0,8 et top-k 40.
SAMPLING = {"temperature": 0.8, "top_k": 40, "max_new_tokens": 100}


def adapter_dir(kind: str) -> Path:
    return HERE / "outputs" / kind / "adapter"


BASE_MODELS = {"base": CONFIG["model"]["name"], "base_360m": "HuggingFaceTB/SmolLM2-360M"}


def load(kind: str, dtype: torch.dtype = torch.float32):
    """Le modèle demandé et son tokenizer, en mode évaluation.

    `kind` : "base" ou "base_360m" (sans adaptation), ou le nom d'un dossier de
    lora_experiments/outputs/ (style, keywords, keywords_360m…). Le modèle de base d'un
    adaptateur est lu dans son adapter_config.json : 135M ou 360M.
    `dtype` : float32 par défaut (bfloat16 n'est pas plus rapide sur une RTX 3050).
    """
    base_name = BASE_MODELS.get(kind)
    if base_name is None:
        path = adapter_dir(kind)
        if not path.exists():
            raise SystemExit(f"adaptateur introuvable ({path}) : lancer d'abord train_lora.py")
        base_name = json.loads((path / "adapter_config.json").read_text())["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    tokenizer.padding_side = "left"  # pour générer en lot, les consignes sont alignées à droite
    tokenizer.pad_token = tokenizer.eos_token  # SmolLM2 n'a pas de token de remplissage (masqué par l'attention)
    model = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
    if kind not in BASE_MODELS:
        model = PeftModel.from_pretrained(model, str(path))
    return model.to(DEVICE).eval(), tokenizer


def make_prompt(kind: str, keywords: list[str]) -> str:
    """La consigne d'entraînement de l'expérience (le modèle de base reçoit celle de l'expérience 2)."""
    if kind.startswith("style"):
        return CONFIG["experiments"]["style"]["prompt"]
    return CONFIG["experiments"]["keywords"]["prompt"].format(keywords=", ".join(keywords))


def cut(text: str) -> str:
    """Les 4 premiers vers, arrêtés à la première ligne vide (comme le modèle historique)."""
    lines = []
    for line in text.split("\n"):
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line.strip())
    return "\n".join(lines[:N_LINES])


class StopAfterLines(StoppingCriteria):
    """Arrête la génération dès que chaque texte du lot a écrit `n` retours à la ligne :
    ce qui suit le 4ᵉ vers serait coupé de toute façon par `cut`."""

    def __init__(self, tokenizer, prompt_length: int, n: int = N_LINES) -> None:
        newline_ids = [i for i in range(len(tokenizer)) if "\n" in tokenizer.decode([i])]
        self.newline_ids = torch.tensor(newline_ids, device=DEVICE)
        self.prompt_length = prompt_length
        self.n = n

    def __call__(self, input_ids: torch.Tensor, scores, **kwargs) -> torch.Tensor:
        written = input_ids[:, self.prompt_length:]
        return torch.isin(written, self.newline_ids).sum(dim=1) >= self.n


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], batch_size: int = 16, stop_after_lines: bool = False) -> list[str]:
    """Une comptine par consigne. La graine se fixe avant l'appel (torch.manual_seed).

    `stop_after_lines` (utilisé par le site) arrête le lot dès que tous les textes ont leurs 4 vers,
    au lieu d'aller jusqu'à max_new_tokens. L'évaluation le laisse à False pour rester reproductible.
    """
    texts = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        stop = StoppingCriteriaList([StopAfterLines(tokenizer, batch["input_ids"].shape[1])]) if stop_after_lines else None
        out = model.generate(**batch, do_sample=True, pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id, stopping_criteria=stop, **SAMPLING)
        new = out[:, batch["input_ids"].shape[1]:]  # seulement ce qui a été écrit après la consigne
        texts += [cut(t) for t in tokenizer.batch_decode(new, skip_special_tokens=True)]
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Écrit des comptines avec SmolLM2 (+ LoRA).")
    parser.add_argument("words", nargs="*", help="mots-clés en anglais (ignorés par le modèle style)")
    parser.add_argument("--model", default="keywords",
                        help="base, style, keywords, ou un autre dossier de outputs/ (ex. keywords_360m)")
    parser.add_argument("--n", type=int, default=1, help="nombre de comptines")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tokenizer = load(args.model)
    words = [w.lower() for w in args.words]
    for text in generate(model, tokenizer, [make_prompt(args.model, words)] * args.n):
        print(text, end="\n\n")


if __name__ == "__main__":
    main()
