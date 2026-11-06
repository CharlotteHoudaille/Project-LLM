"""Compare les modèles sur le jeu de test (jamais vu à l'entraînement, ni par le modèle historique).

    uv run --group lora python lora_experiments/evaluate.py
    uv run --group lora python lora_experiments/evaluate.py --smoke    # avec les adaptateurs du test rapide

Modèles comparés, avec les mêmes consignes, la même graine et les mêmes réglages de tirage :
1. base     : SmolLM2-135M sans adaptation ;
2. style    : SmolLM2 + LoRA, expérience 1 ;
3. keywords : SmolLM2 + LoRA, expérience 2 ;
4. historique : notre GPT de 8,4 M (notebook 08), le modèle seul (sans coup de pouce ni meilleur de N) ;
5. si l'adaptateur existe : base_360m (SmolLM2-360M sans adaptation) et keywords_360m (expérience 2 sur 360M).

Mesures :
- bits par caractère sur les comptines de test : la loss totale de la comptine / son nombre de
  caractères / ln 2. Contrairement à la perplexité, elle ne dépend pas du tokenizer : c'est la
  seule mesure de loss qui compare SmolLM2 et notre modèle historique ;
- perplexité de test (entre modèles SmolLM2 seulement, qui partagent le même tokenizer) ;
- sur les comptines écrites : mots-clés repris, comptines avec tous les mots, vers qui riment
  (tout schéma), part de comptines AABB, 4 vers, mots par vers, diversité (distinct-2) et vers
  recopiés du corpus d'entraînement.

Sorties dans lora_experiments/results/ : metrics.json, examples.md, comparison.png, losses.png.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import comptines  # noqa: E402  (le modèle historique)
from generate_lora import DEVICE, generate, load, make_prompt  # noqa: E402
from prepare_data import letters, scheme  # noqa: E402

CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
DATA = ROOT / CONFIG["data"]["out_dir"]
RESULTS = HERE / "results"
DEMO = [["cat", "moon"], ["king", "pie"], ["rain", "garden"], ["dog", "bone", "star"], ["mouse", "cheese"],
        ["sun", "sea", "boat"]]


def read(name: str) -> list[dict]:
    with open(DATA / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --- Bits par caractère ------------------------------------------------------------------------


@torch.no_grad()
def bpc_smollm(model, tokenizer, prompts: list[str], poems: list[str]) -> tuple[float, float]:
    """Bits par caractère des comptines (fin de texte comprise) et perplexité par token."""
    total_nll, total_tokens = 0.0, 0
    for prompt, poem in zip(prompts, poems):
        p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        r = tokenizer(poem, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        ids = torch.tensor([p + r], device=DEVICE)
        labels = torch.tensor([[-100] * len(p) + r], device=DEVICE)
        loss = model(input_ids=ids, labels=labels).loss.item()  # moyenne sur les tokens de la comptine
        total_nll += loss * len(r)
        total_tokens += len(r)
    chars = sum(len(poem) for poem in poems)
    return total_nll / chars / math.log(2), math.exp(total_nll / total_tokens)


@torch.no_grad()
def bpc_historic(model, tokenizer, keyword_sets: list[list[str]], poems: list[str]) -> float:
    """Même mesure pour le modèle historique, avec sa propre consigne (« rhyme: … »)."""
    total_nll = 0.0
    for keywords, poem in zip(keyword_sets, poems):
        p = tokenizer.encode(comptines.make_prompt(keywords, "rhyme"))
        ids = p + tokenizer.encode(poem) + [tokenizer.eot]
        x = torch.tensor([ids[:-1]], device=DEVICE)
        y = torch.tensor([ids[1:]], device=DEVICE)
        y[0, : len(p) - 1] = -1  # la consigne n'est pas notée
        _, loss = model(x, y)
        total_nll += loss.item() * int((y != -1).sum())
    return total_nll / sum(len(poem) for poem in poems) / math.log(2)


def load_historic():
    ck = torch.load(ROOT / "checkpoints" / "gpt_comptines_v2.pt", map_location=DEVICE)
    model = comptines.GPTLanguageModel(**ck["config"]).to(DEVICE).eval()
    model.load_state_dict(ck["model_state"])
    return model, comptines.BPETokenizer(ck["gpt2_ids"])


# --- Mesures sur les comptines écrites -------------------------------------------------------------


def describe(texts: list[str], keyword_sets: list[list[str]], train_lines: set[str]) -> dict:
    n = len(texts)
    lines = [[l for l in t.split("\n") if l.strip()] for t in texts]
    words = [re.findall(r"[a-z']+", t.lower()) for t in texts]
    bigrams = [(a, b) for w in words for a, b in zip(w, w[1:])]
    gen_lines = [letters(l) for ls in lines for l in ls if len(l.split()) >= 3]
    return {
        "keywords_used": sum(comptines.keyword_hits(k, t) for k, t in zip(keyword_sets, texts)) / n,
        "all_keywords": sum(comptines.keyword_hits(k, t) == 1 for k, t in zip(keyword_sets, texts)) / n,
        "rhyming_lines": sum(comptines.rhyme_rate(t) for t in texts) / n,
        "aabb": sum(scheme(t) == "AABB" for t in texts) / n,
        "four_lines": sum(len(l) == 4 for l in lines) / n,
        "words_per_line": sum(len(x.split()) for l in lines for x in l) / max(1, sum(map(len, lines))),
        "distinct_2": len(set(bigrams)) / max(1, len(bigrams)),
        "copied_lines": sum(l in train_lines for l in gen_lines) / max(1, len(gen_lines)),
    }


# --- Graphiques -------------------------------------------------------------------------------


def plot_comparison(results: dict, path: Path) -> None:
    names = list(results)
    panels = [("keywords_used", "mots-clés repris (%)", 100), ("rhyming_lines", "vers qui riment (%)", 100),
              ("bpc", "bits par caractère (plus bas = mieux)", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, title, scale) in zip(axes, panels):
        values = [results[n][key] * scale for n in names]
        palette = ["#9aa5b1", "#299bde", "#389975", "#c3c9d0", "#1f6f50", "#f7914e"]
        ax.bar(names, values, color=palette[: len(names)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.1f}" if scale == 100 else f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_losses(runs: dict, path: Path) -> None:
    plt.figure(figsize=(8, 3.8))
    for name, metrics in runs.items():
        history = metrics["log_history"]
        train = [(h["step"], h["loss"]) for h in history if "loss" in h]
        val = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
        line = plt.plot(*zip(*train), alpha=0.5, label=f"{name} : entraînement")[0]
        plt.plot(*zip(*val), marker="o", color=line.get_color(), label=f"{name} : validation")
    plt.xlabel("pas")
    plt.ylabel("loss (par token)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="adaptateurs du test rapide, 16 exemples")
    args = parser.parse_args()
    suffix = "_smoke" if args.smoke else ""
    RESULTS.mkdir(exist_ok=True)

    test = read("keywords_test")
    if args.smoke:
        test = test[:16]
    keyword_sets = [e["keywords"] for e in test]
    poems = [e["response"] for e in test]
    train_lines = {letters(l) for e in read("style_train") for l in e["response"].split("\n")}
    seed = CONFIG["seed"]

    # Les variantes 360M ne sont évaluées que si leur adaptateur existe (et pas en test rapide).
    kinds = ["base", "style", "keywords"]
    if not args.smoke and (HERE / "outputs" / "keywords_360m" / "adapter").exists():
        kinds += ["base_360m", "keywords_360m"]
    results, examples, generations = {}, {}, {}
    for kind in kinds:
        label = kind if kind.startswith("base") else kind + suffix
        model, tokenizer = load(label)
        prompts = [make_prompt(kind, k) for k in keyword_sets]
        bpc, ppl = bpc_smollm(model, tokenizer, prompts, poems)
        torch.manual_seed(seed)
        texts = generate(model, tokenizer, prompts)
        results[kind] = {"bpc": bpc, "perplexity": ppl, **describe(texts, keyword_sets, train_lines)}
        generations[kind] = texts
        torch.manual_seed(seed)
        examples[kind] = generate(model, tokenizer, [make_prompt(kind, k) for k in DEMO])
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    model, tokenizer = load_historic()
    bpc = bpc_historic(model, tokenizer, keyword_sets, poems)
    torch.manual_seed(seed)
    texts = comptines.write_texts(model, tokenizer, keyword_sets, n_lines=4, max_batch=64, genre="rhyme")
    results["historique"] = {"bpc": bpc, "perplexity": None, **describe(texts, keyword_sets, train_lines)}
    generations["historique"] = texts
    torch.manual_seed(seed)
    examples["historique"] = comptines.write_texts(model, tokenizer, DEMO, n_lines=4, genre="rhyme")
    results["vraies comptines"] = {"bpc": None, "perplexity": None, **describe(poems, keyword_sets, train_lines)}

    runs = {}
    for kind in ("style", "keywords", "keywords_360m"):
        path = HERE / "outputs" / (kind + suffix) / "metrics.json"
        if path.exists():
            runs[kind] = json.loads(path.read_text(encoding="utf-8"))
    training = {k: {key: m[key] for key in ("trainable_params", "total_params", "train_seconds", "steps",
                                             "best_step", "val_loss", "val_perplexity", "test_loss",
                                             "test_perplexity", "device")} for k, m in runs.items()}

    (RESULTS / f"metrics{suffix}.json").write_text(
        json.dumps({"test_examples": len(test), "generation": results, "training": training}, indent=2),
        encoding="utf-8")
    # Toutes les comptines écrites sur le jeu de test, pour les relire.
    (RESULTS / f"generations{suffix}.json").write_text(json.dumps(
        [{"prompt_keywords": k, "reference": p, **{m: generations[m][i] for m in generations}}
         for i, (k, p) in enumerate(zip(keyword_sets, poems))], indent=2, ensure_ascii=False), encoding="utf-8")
    with open(RESULTS / f"examples{suffix}.md", "w", encoding="utf-8") as f:
        f.write("# Comptines écrites par chaque modèle (mêmes consignes, même graine)\n\n")
        for i, words in enumerate(DEMO):
            f.write(f"## {', '.join(words)}\n\n")
            for name, texts in examples.items():
                note = " (ne reçoit pas les mots-clés)" if name == "style" else ""
                f.write(f"**{name}**{note}\n\n```\n{texts[i]}\n```\n\n")
    plot_comparison({k: v for k, v in results.items() if v["bpc"] is not None}, RESULTS / f"comparison{suffix}.png")
    if runs:
        plot_losses(runs, RESULTS / f"losses{suffix}.png")

    columns = ["bpc", "perplexity", "keywords_used", "all_keywords", "rhyming_lines", "aabb", "four_lines",
               "words_per_line", "distinct_2", "copied_lines"]
    print(f"jeu de test : {len(test)} comptines de 4 vers\n")
    print(f"{'':17}" + "".join(f"{c:>15}" for c in columns))
    for name, r in results.items():
        cells = []
        for c in columns:
            v = r[c]
            cells.append(f"{'—':>15}" if v is None else f"{v:>15.2f}" if c in ("bpc", "perplexity", "words_per_line")
                         else f"{v:>15.1%}")
        print(f"{name:17}" + "".join(cells))
    for k, t in training.items():
        print(f"\n{k} : {t['trainable_params']:,} paramètres entraînables sur {t['total_params']:,}, "
              f"{t['train_seconds']:.0f} s, meilleur pas {t['best_step']}/{t['steps']}, "
              f"val {t['val_loss']:.3f} (perplexité {t['val_perplexity']:.1f})")
    print(f"\nrésultats → {RESULTS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
