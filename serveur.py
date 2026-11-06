"""Le site du générateur de comptines, en local.

Sert la page du dossier `site/` et répond à `POST /api/generate` avec une comptine
écrite par l'un des deux modèles du projet :

- "original" : notre GPT, construit et entraîné de zéro (notebooks 01 à 08), comme `generer.py` ;
- "lora"     : l'expérience LoRA (lora_experiments/) : SmolLM2 pré-entraîné + adaptateur LoRA.

    uv run python serveur.py    → les deux modèles (le groupe de dépendances "lora" est installé par défaut)
    → http://localhost:8000

`GET /api/models` dit quels modèles sont disponibles. Le serveur n'utilise que la bibliothèque
standard (`http.server`) en plus du projet.
"""

import json
import os
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from comptines import BPETokenizer, GPTLanguageModel, best_of_n, keyword_hits, poem_score, rhyme_rate

ROOT = Path(__file__).resolve().parent
SITE_DIR = ROOT / "site"
CHECKPOINTS = [ROOT / "checkpoints" / "gpt_comptines_v2.pt", ROOT / "checkpoints" / "gpt_comptines_final.pt"]
# En local : 127.0.0.1:8000. Sur Hugging Face Spaces, le Dockerfile fixe HOST=0.0.0.0 et PORT=7860.
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 8000))
MAX_WORDS = 3
LORA_ADAPTER = os.environ.get("LORA_ADAPTER", "keywords_360m")  # dossier de lora_experiments/outputs/

# --- Le modèle, chargé une seule fois au démarrage ----------------------------------

checkpoint_path = next((p for p in CHECKPOINTS if p.exists()), None)
if checkpoint_path is None:
    raise SystemExit("aucun modèle trouvé : exécuter d'abord 08_finetuning_melange.ipynb")
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
checkpoint = torch.load(checkpoint_path, map_location=device)
settings = {"keyword_boost": 0.0, "n": 8, "keyword_weight": 1.0, "max_words": None, "genre": "words",
            **checkpoint.get("generation", {})}
tokenizer = BPETokenizer(checkpoint["gpt2_ids"])
model = GPTLanguageModel(**checkpoint["config"]).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
generate_lock = threading.Lock()  # une génération à la fois sur le GPU


def write_comptine(words: list[str]) -> dict:
    """Une comptine de 4 vers (meilleur de N, coup de pouce), avec les mots repris."""
    with generate_lock:
        start = time.perf_counter()
        text, _ = best_of_n(model, tokenizer, words, n=settings["n"], keyword_weight=settings["keyword_weight"],
                            max_words=settings["max_words"], keyword_boost=settings["keyword_boost"],
                            genre=settings["genre"])
    return describe(text, words, start, "original")


def describe(text: str, words: list[str], start: float, model_id: str) -> dict:
    return {
        "model": model_id,
        "lines": text.split("\n"),
        "words": words,
        "found": [w for w in words if keyword_hits([w], text)],
        "rhyme_rate": rhyme_rate(text),
        "seconds": round(time.perf_counter() - start, 2),
    }


# --- L'expérience LoRA, chargée en arrière-plan si ses bibliothèques sont là ----------------

lora = {"model": None, "tokenizer": None, "module": None, "status": "loading", "reason": ""}


def load_lora() -> None:
    """Charge SmolLM2 + l'adaptateur LoRA. Sans le groupe de dépendances "lora", le site
    fonctionne quand même : seul le modèle original est proposé."""
    try:
        sys.path.insert(0, str(ROOT / "lora_experiments"))
        import generate_lora  # transformers, peft : groupe "lora" de pyproject.toml
        model_lora, tokenizer_lora = generate_lora.load(LORA_ADAPTER)
        lora.update(model=model_lora, tokenizer=tokenizer_lora, module=generate_lora, status="ready")
    except ImportError:
        lora.update(status="unavailable", reason="LoRA libraries are not installed: run \"uv sync\" (the lora group is "
                                                  "installed by default), then restart the server.")
    except SystemExit as err:  # adaptateur pas encore entraîné
        lora.update(status="unavailable", reason=f"No LoRA adapter yet ({err}).")


def write_lora(words: list[str]) -> dict:
    """Une comptine de SmolLM2 + LoRA : 8 candidates, on garde la meilleure (même score que l'original)."""
    g = lora["module"]
    with generate_lock:
        start = time.perf_counter()
        candidates = g.generate(lora["model"], lora["tokenizer"], [g.make_prompt("keywords", words)] * settings["n"],
                                stop_after_lines=True)
    best = max(candidates, key=lambda t: poem_score(words, t, 4, settings["keyword_weight"], settings["max_words"]))
    return describe(best, words, start, "lora")


def models_info() -> dict:
    return {"original": {"available": True},
            "lora": {"available": lora["status"] == "ready", "status": lora["status"], "reason": lora["reason"],
                     "adapter": LORA_ADAPTER}}


# --- Le serveur ----------------------------------------------------------------------


class Handler(SimpleHTTPRequestHandler):
    """Fichiers de `site/` en GET, génération en POST /api/generate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_DIR), **kwargs)

    def do_OPTIONS(self) -> None:
        """Pré-vérification CORS du navigateur, avant un POST venant d'une page ouverte en fichier."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_cors_headers()
        self.end_headers()

    def send_cors_headers(self) -> None:
        # La page peut être ouverte directement depuis le disque (file://) : on l'autorise à appeler l'API.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_GET(self) -> None:
        if self.path == "/api/models":
            self.send_json(models_info())
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            words = [re.sub(r"[^a-z]", "", str(w).lower())[:20] for w in body.get("words", [])]
            words = list(dict.fromkeys(w for w in words if w))[:MAX_WORDS]  # sans doublon, 3 au plus
        except (ValueError, TypeError):
            self.send_json({"error": "requête illisible"}, HTTPStatus.BAD_REQUEST)
            return
        if not words:
            self.send_json({"error": "donne au moins un mot"}, HTTPStatus.BAD_REQUEST)
            return
        if body.get("model") == "lora":
            if lora["status"] != "ready":
                message = "the LoRA model is still loading" if lora["status"] == "loading" else lora["reason"]
                self.send_json({"error": message}, HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self.send_json(write_lora(words))
        else:
            self.send_json(write_comptine(words))

    def send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        if self.path.startswith("/api/"):
            super().log_message(format, *args)


if __name__ == "__main__":
    print(f"modèle : {checkpoint_path.name} sur {device}")
    threading.Thread(target=load_lora, daemon=True).start()  # le site répond pendant ce temps
    print(f"site   : http://localhost:{PORT}  (LoRA : chargement de {LORA_ADAPTER} en arrière-plan)")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
