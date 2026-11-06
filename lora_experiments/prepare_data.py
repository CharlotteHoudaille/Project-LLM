"""Prépare le corpus de comptines pour le fine-tuning LoRA de SmolLM2.

    uv run --group lora python lora_experiments/prepare_data.py

Étapes (toutes réglées par config.yaml) :

1. chargement des comptines du projet (notebook 01), en notant d'où vient chacune
   (l'entraînement ou la validation du modèle historique) ;
2. nettoyage léger (espaces, lignes vides) ;
3. suppression des doublons exacts ;
4. regroupement des variantes : deux comptines qui partagent leur premier ou leur dernier
   vers sont la même comptine sous deux formes, et restent du même côté de la séparation ;
5. mots-clés tirés de la comptine elle-même (la consigne ne dit jamais quelque chose de faux
   sur le poème) et schéma de rimes détecté (AABB, ABAB, ABCB ou libre), gardé pour l'analyse ;
6. séparation entraînement / validation / test **par groupe**, sans fuite ;
7. formatage des exemples des deux expériences (consigne → réponse) ;
8. tokenisation avec le tokenizer de SmolLM2, et labels à -100 sur la consigne.

Sorties dans lora_experiments/data/ : un fichier JSONL lisible par expérience et par split,
les datasets tokenisés (format Hugging Face) et un résumé chiffré (summary.json).
"""

import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from comptines import content_words, last_word, rhyme_key, sample_keywords  # noqa: E402  (code du projet)

CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))

# --- 1-3. Chargement, nettoyage, doublons ------------------------------------------------


def clean(poem: str) -> str:
    """Espaces normalisés dans chaque vers, vers vides retirés ; l'ordre des vers est gardé."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in poem.split("\n")]
    return "\n".join(line for line in lines if line)


def letters(text: str) -> str:
    """Le texte réduit à ses mots en minuscules : la clé de comparaison des doublons."""
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def load_poems() -> list[dict]:
    poems, seen = [], set()
    for source in CONFIG["data"]["sources"]:
        origin = "historic_train" if "train" in source else "historic_val"
        for raw in (ROOT / source).read_text(encoding="utf-8").split("\n\n"):
            poem = clean(raw)
            key = letters(poem)
            if poem.count("\n") < 1 or key in seen:  # au moins 2 vers, pas de doublon exact
                continue
            seen.add(key)
            poems.append({"id": len(poems), "text": poem, "origin": origin, "n_lines": poem.count("\n") + 1})
    return poems


# --- 4. Groupes de variantes ------------------------------------------------------------


def group_variants(poems: list[dict]) -> None:
    """Union-find : deux comptines qui partagent un premier ou un dernier vers (d'au moins
    3 mots) tombent dans le même groupe. Ajoute le champ `group` à chaque comptine."""
    parent = list(range(len(poems)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict[str, int] = {}
    for p in poems:
        lines = p["text"].split("\n")
        # Une liste et non un set : l'ordre de parcours doit être le même à chaque lancement,
        # sinon les groupes (et donc les splits) changeraient d'une exécution à l'autre.
        for key in ("first:" + letters(lines[0]), "last:" + letters(lines[-1])):
            if len(key.split(":", 1)[1].split()) < 3:
                continue
            if key in owner:
                parent[find(p["id"])] = find(owner[key])
            else:
                owner[key] = p["id"]
    for p in poems:
        p["group"] = find(p["id"])


# --- 5. Schéma de rimes et mots-clés -----------------------------------------------------


def rhymes(a: str, b: str) -> bool:
    """Deux vers riment : même fin de mot, et pas le même mot (comptines.py)."""
    return bool(rhyme_key(a)) and rhyme_key(a) == rhyme_key(b) and last_word(a) != last_word(b)


def scheme(poem: str) -> str:
    """Le schéma de rimes d'une comptine de 4 vers ; « free » sinon."""
    lines = poem.split("\n")
    if len(lines) != 4:
        return "free"
    if rhymes(lines[0], lines[1]) and rhymes(lines[2], lines[3]):
        return "AABB"
    if rhymes(lines[0], lines[2]) and rhymes(lines[1], lines[3]):
        return "ABAB"
    if rhymes(lines[1], lines[3]):
        return "ABCB"
    return "free"


def keyword_candidates(text: str) -> list[str]:
    """Les mots porteurs de sens (comptines.content_words), sans les morceaux de contractions :
    « didn't » donnerait sinon le mot-clé « didn »."""
    lower = text.lower()
    return [w for w in content_words(text) if not re.search(rf"\b{w}'[a-z]", lower)]


def compute_idf(poems: list[dict]) -> dict[str, float]:
    """IDF de chaque mot sur les comptines d'entraînement seulement (rien n'est appris du test)."""
    df = Counter(w for p in poems for w in keyword_candidates(p["text"]))
    n = len(poems)
    return {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}


# --- 6. Séparation par groupes -------------------------------------------------------------


def split(poems: list[dict]) -> None:
    """test : les groupes vus seulement dans la validation du modèle historique (jamais vus par lui,
    ni sous une autre forme) ; validation LoRA : une part des autres groupes ; le reste : entraînement."""
    groups: dict[int, list[dict]] = {}
    for p in poems:
        groups.setdefault(p["group"], []).append(p)
    rng = random.Random(CONFIG["seed"])
    trainable = sorted(g for g, members in groups.items() if any(m["origin"] == "historic_train" for m in members))
    val_groups = set(rng.sample(trainable, k=round(CONFIG["data"]["val_share"] * len(trainable))))
    for g, members in groups.items():
        name = "test" if g not in trainable else "validation" if g in val_groups else "train"
        for m in members:
            m["split"] = name


# --- 7. Les exemples des deux expériences ---------------------------------------------------------


def build_examples(poems: list[dict], idf: dict[str, float]) -> dict[str, dict[str, list[dict]]]:
    exp = CONFIG["experiments"]
    out = {"style": {}, "keywords": {}}
    for split_name in ("train", "validation", "test"):
        part = [p for p in poems if p["split"] == split_name]
        out["style"][split_name] = [
            {"id": p["id"], "group": p["group"], "prompt": exp["style"]["prompt"], "response": p["text"],
             "scheme": p["scheme"], "n_lines": p["n_lines"]}
            for p in part
        ]
        examples = []
        n_sets = CONFIG["data"]["keyword_sets_train"] if split_name == "train" else 1
        for p in part:
            words = keyword_candidates(p["text"])
            if p["n_lines"] != 4 or not words:
                continue
            for variant in range(n_sets):
                rng = random.Random(CONFIG["seed"] * 1000 + p["id"] * 10 + variant)  # reproductible
                keywords = sample_keywords(words, rng, CONFIG["data"]["max_keywords"], idf=idf)
                prompt = exp["keywords"]["prompt"].format(keywords=", ".join(keywords))
                examples.append({"id": p["id"], "group": p["group"], "prompt": prompt, "response": p["text"],
                                 "keywords": keywords, "scheme": p["scheme"], "n_lines": 4, "variant": variant})
        out["keywords"][split_name] = examples
    return out


# --- 8. Tokenisation --------------------------------------------------------------------------


def tokenize(examples: list[dict], tokenizer, max_length: int) -> tuple[Dataset, int]:
    """input_ids = consigne + réponse + <|endoftext|> ; labels = -100 sur la consigne.
    La loss ne porte donc que sur la comptine, comme dans le projet historique."""
    rows, truncated = [], 0
    for ex in examples:
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(ex["response"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        ids = (prompt_ids + response_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
        truncated += len(prompt_ids) + len(response_ids) > max_length
        rows.append({"input_ids": ids, "labels": labels, "id": ex["id"]})
    return Dataset.from_list(rows), truncated


def main() -> None:
    out_dir = ROOT / CONFIG["data"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    poems = load_poems()
    group_variants(poems)
    for p in poems:
        p["scheme"] = scheme(p["text"])
    split(poems)
    idf = compute_idf([p for p in poems if p["split"] == "train"])
    data = build_examples(poems, idf)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["name"])
    summary = {"poems": len(poems), "groups": len({p["group"] for p in poems}),
               "splits": {s: sum(p["split"] == s for p in poems) for s in ("train", "validation", "test")},
               "schemes_4_lines_train": Counter(p["scheme"] for p in poems
                                                if p["split"] == "train" and p["n_lines"] == 4),
               "experiments": {}}
    for exp_name, splits in data.items():
        tokenized, stats = {}, {}
        for split_name, examples in splits.items():
            with open(out_dir / f"{exp_name}_{split_name}.jsonl", "w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            ds, truncated = tokenize(examples, tokenizer, CONFIG["data"]["max_length"])
            tokenized[split_name] = ds
            lengths = sorted(len(r) for r in ds["input_ids"])
            stats[split_name] = {"examples": len(examples), "tokens_median": lengths[len(lengths) // 2],
                                 "tokens_max": lengths[-1], "truncated": truncated}
        DatasetDict(tokenized).save_to_disk(str(out_dir / f"tokenized_{exp_name}"))
        summary["experiments"][exp_name] = stats
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
