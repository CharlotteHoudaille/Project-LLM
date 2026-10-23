"""Écrit une comptine de 4 vers à partir de mots-clés, avec le modèle final.

Utilise le modèle du notebook 08 (`gpt_comptines_v2.pt`) s'il existe, sinon celui
du notebook 07 (`gpt_comptines_final.pt`).

Exemples :
    uv run python generer.py cat moon
    uv run python generer.py king pie --n 16 --temperature 0.9
    uv run python generer.py moon cat dog --boost 0      # sans coup de pouce : le modèle seul
    uv run python generer.py dog bone --candidates      # affiche les N candidates
"""

import argparse
from pathlib import Path

import torch

from comptines import BPETokenizer, GPTLanguageModel, best_of_n, keyword_hits, poem_score, rhyme_rate

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINTS = [CHECKPOINT_DIR / "gpt_comptines_v2.pt", CHECKPOINT_DIR / "gpt_comptines_final.pt"]


def main() -> None:
    checkpoint_path = next((p for p in CHECKPOINTS if p.exists()), None)
    if checkpoint_path is None:
        raise SystemExit("aucun modèle trouvé : exécuter d'abord 07_finetuning_comptines.ipynb ou 08_finetuning_melange.ipynb")
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    defaults = {"keyword_boost": 0.0, "n": 8, "keyword_weight": 1.0, "max_words": None, "genre": "words",
                **checkpoint.get("generation", {})}  # le modèle 07 n'a pas de réglages enregistrés

    parser = argparse.ArgumentParser(description="Écrit une comptine de 4 vers à partir de mots-clés.")
    parser.add_argument("words", nargs="+", help="les mots-clés, en anglais (ex. : cat moon)")
    parser.add_argument("--n", type=int, default=defaults["n"], help="nombre de comptines candidates (meilleur de N)")
    parser.add_argument("--boost", type=float, default=defaults["keyword_boost"],
                        help="coup de pouce sur les mots-clés pas encore écrits (0 : aucun)")
    parser.add_argument("--temperature", type=float, default=0.8, help="< 1 : plus sûr, > 1 : plus varié")
    parser.add_argument("--top-k", type=int, default=40, help="ne tire que parmi les k tokens les plus probables")
    parser.add_argument("--seed", type=int, default=None, help="graine, pour retrouver le même résultat")
    parser.add_argument("--candidates", action="store_true", help="affiche aussi toutes les candidates")
    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = BPETokenizer(checkpoint["gpt2_ids"])
    model = GPTLanguageModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    words = [w.lower() for w in args.words]
    weight, max_words = defaults["keyword_weight"], defaults["max_words"]
    best, candidates = best_of_n(model, tokenizer, words, n=args.n, keyword_weight=weight, max_words=max_words,
                                 temperature=args.temperature, top_k=args.top_k, keyword_boost=args.boost,
                                 genre=defaults["genre"])

    if args.candidates:
        score = lambda t: poem_score(words, t, keyword_weight=weight, max_words=max_words)  # noqa: E731
        for text in sorted(candidates, key=score, reverse=True):
            print(f"score {score(text):5.2f}\n   " + text.replace("\n", "\n   ") + "\n")
        print("=" * 40)

    found = [w for w in words if keyword_hits([w], best)]
    print(best)
    print(f"\n(modèle : {checkpoint_path.name}   mots repris : {', '.join(found) or 'aucun'}   "
          f"vers qui riment : {rhyme_rate(best):.0%})")


if __name__ == "__main__":
    main()
