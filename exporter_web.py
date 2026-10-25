"""Exporte le modèle final pour qu'il tourne dans le navigateur, sans serveur.

    uv run python exporter_web.py

Produit le dossier `docs/`, que GitHub Pages sert tel quel :

- `docs/model/weights.bin` : les poids en demi-précision (float16), à la suite ;
- `docs/model/manifest.json` : la configuration, les réglages de génération et,
  pour chaque tenseur, sa forme et sa position dans `weights.bin` ;
- `docs/model/tokenizer.json` : le vocabulaire réduit (notebook 04) et la table
  du BPE de GPT-2, pour encoder et décoder en JavaScript ;
- la page du site (copiée de `site/`), en mode « navigateur » : le modèle tourne
  dans `site/model.js`, la version JavaScript du GPT de `comptines.py`.

La version Python (`serveur.py`, `generer.py`) reste la référence et ne change pas.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from comptines import BPETokenizer

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "gpt_comptines_v2.pt"
SITE_DIR = ROOT / "site"
DOCS_DIR = ROOT / "docs"
MODEL_DIR = DOCS_DIR / "model"
SITE_FILES = ["index.html", "style.css", "app.js", "model.js", "worker.js"]


def model_tensors(state: dict, config: dict) -> list[tuple[str, torch.Tensor]]:
    """Les tenseurs dans l'ordre et la forme que lit `model.js`.

    Les têtes d'attention d'un bloc sont empilées : la ligne `h * head_size + j`
    de `wq` est la ligne `j` de la tête `h`, comme le `torch.cat` de `MultiHeadAttention`.
    La couche de sortie partage la matrice de l'embedding (weight tying) : seul son biais est exporté.
    """
    tensors = [("tok_emb", state["token_embedding_table.weight"]),
               ("pos_emb", state["position_embedding_table.weight"])]
    for i in range(config["n_layer"]):
        p = f"blocks.{i}."
        heads = range(config["n_head"])
        tensors += [
            (f"l{i}.ln1_w", state[p + "ln1.weight"]), (f"l{i}.ln1_b", state[p + "ln1.bias"]),
            (f"l{i}.wq", torch.cat([state[p + f"sa.heads.{h}.query.weight"] for h in heads])),
            (f"l{i}.wk", torch.cat([state[p + f"sa.heads.{h}.key.weight"] for h in heads])),
            (f"l{i}.wv", torch.cat([state[p + f"sa.heads.{h}.value.weight"] for h in heads])),
            (f"l{i}.wproj", state[p + "sa.proj.weight"]), (f"l{i}.bproj", state[p + "sa.proj.bias"]),
            (f"l{i}.ln2_w", state[p + "ln2.weight"]), (f"l{i}.ln2_b", state[p + "ln2.bias"]),
            (f"l{i}.w1", state[p + "ffwd.net.0.weight"]), (f"l{i}.b1", state[p + "ffwd.net.0.bias"]),
            (f"l{i}.w2", state[p + "ffwd.net.2.weight"]), (f"l{i}.b2", state[p + "ffwd.net.2.bias"]),
        ]
    tensors += [("ln_f_w", state["ln_f.weight"]), ("ln_f_b", state["ln_f.bias"]),
                ("lm_bias", state["lm_head.bias"])]
    return tensors


def main() -> None:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    config, state = checkpoint["config"], checkpoint["model_state"]
    tokenizer = BPETokenizer(checkpoint["gpt2_ids"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Les poids, en float16 à la suite, et leur table des matières.
    entries, chunks, offset = [], [], 0
    for name, tensor in model_tensors(state, config):
        array = tensor.detach().to(torch.float16).numpy().ravel()
        entries.append({"name": name, "shape": list(tensor.shape), "offset": offset})
        chunks.append(array)
        offset += array.size
    weights = np.concatenate(chunks).astype("<f2")
    (MODEL_DIR / "weights.bin").write_bytes(weights.tobytes())

    generation = {"keyword_boost": 0.0, "n": 8, "keyword_weight": 1.0, "max_words": None, "genre": "words",
                  "temperature": 0.8, "top_k": 40, "n_lines": 4, "max_new_tokens": 250,
                  **checkpoint.get("generation", {})}
    manifest = {"config": {k: config[k] for k in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer")},
                "generation": generation, "dtype": "float16", "tensors": entries}
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # 2. Le tokenizer : pour GPT-2, le rang d'un token BPE est son identifiant.
    #    Chaque token est écrit comme une chaîne latin-1 (un caractère par octet).
    ranks = sorted(tokenizer.enc._mergeable_ranks.items(), key=lambda kv: kv[1])
    assert [r for _, r in ranks] == list(range(len(ranks)))
    tok = {"gpt2_ids": tokenizer.gpt2_ids, "eot": tokenizer.eot, "eot_gpt2": tokenizer.enc.eot_token,
           "ranks": [b.decode("latin-1") for b, _ in ranks]}
    (MODEL_DIR / "tokenizer.json").write_text(json.dumps(tok, separators=(",", ":")))

    # 3. La page, en mode navigateur.
    for name in SITE_FILES:
        shutil.copy2(SITE_DIR / name, DOCS_DIR / name)
    index = (DOCS_DIR / "index.html").read_text(encoding="utf-8")
    index = index.replace('<script src="app.js"></script>',
                          '<script>window.RHYME_MODE = "browser";</script>\n  <script src="app.js"></script>')
    (DOCS_DIR / "index.html").write_text(index, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("")  # GitHub Pages : servir les fichiers tels quels

    n_params = sum(e["shape"][0] * (e["shape"][1] if len(e["shape"]) > 1 else 1) for e in entries)
    print(f"{len(entries)} tenseurs, {n_params:,} paramètres → docs/model/weights.bin "
          f"({weights.nbytes / 2**20:.1f} Mo)")
    print(f"tokenizer : {len(tok['ranks']):,} tokens GPT-2, vocabulaire réduit {len(tok['gpt2_ids']):,}")
    print(f"site → {DOCS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
