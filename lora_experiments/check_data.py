"""Vérifie le corpus préparé par prepare_data.py et en montre des exemples.

    uv run --group lora python lora_experiments/check_data.py            # vérifications + 2 exemples
    uv run --group lora python lora_experiments/check_data.py --show 5   # plus d'exemples

Vérifications :
- aucun groupe de variantes à cheval sur deux splits (pas de fuite vers la validation ou le test) ;
- chaque mot-clé de l'expérience 2 apparaît bien dans sa comptine ;
- dans les données tokenisées, la consigne est masquée (-100) et la comptine finit par <|endoftext|>.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
DATA = HERE.parent / CONFIG["data"]["out_dir"]
SPLITS = ("train", "validation", "test")


def read(name: str) -> list[dict]:
    with open(DATA / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=2, help="nombre d'exemples à afficher par expérience")
    args = parser.parse_args()
    ok = True

    style = {s: read(f"style_{s}") for s in SPLITS}
    keywords = {s: read(f"keywords_{s}") for s in SPLITS}

    where = defaultdict(set)
    for s in SPLITS:
        for row in style[s]:
            where[row["group"]].add(s)
    leaks = sum(len(v) > 1 for v in where.values())
    print(f"groupes de variantes : {len(where)}   à cheval sur deux splits : {leaks}")
    ok &= leaks == 0

    all_kw = [e for s in SPLITS for e in keywords[s]]
    present = sum(all(k in set(re.findall(r"[a-z]+", e["response"].lower())) for k in e["keywords"]) for e in all_kw)
    print(f"mots-clés présents dans leur comptine : {present}/{len(all_kw)}")
    ok &= present == len(all_kw)

    print("schémas de rimes des comptines de 4 vers (entraînement) :",
          dict(Counter(e["scheme"] for e in keywords["train"] if e["variant"] == 0)))

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["name"])
    for exp in ("style", "keywords"):
        ds = load_from_disk(str(DATA / f"tokenized_{exp}"))["train"]
        bad = 0
        for row in ds:
            first = next(i for i, label in enumerate(row["labels"]) if label != -100)
            bad += row["labels"][-1] != tokenizer.eos_token_id or any(l != -100 for l in row["labels"][:first])
        print(f"{exp:8} : {len(ds)} exemples tokenisés, mal formés : {bad}")
        ok &= bad == 0

    rng = random.Random(0)
    for exp, rows in (("style", style["train"]), ("keywords", keywords["train"])):
        print(f"\n===== Exemples d'entraînement — {exp} =====")
        for ex in rng.sample(rows, args.show):
            print(f"--- consigne (non notée) ---\n{ex['prompt']}--- réponse (notée) ---\n{ex['response']}<|endoftext|>\n")

    print("TOUT EST CONFORME" if ok else "PROBLÈMES DÉTECTÉS")


if __name__ == "__main__":
    main()
