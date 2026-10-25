"""Le site du générateur de comptines, en local.

Sert la page du dossier `site/` et répond à `POST /api/generate` avec une
comptine écrite par le modèle final (`generer.py` en version web).

    uv run python serveur.py
    → http://localhost:8000

Uniquement la bibliothèque standard (`http.server`) en plus du projet : rien à installer.
"""

import json
import os
import re
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from comptines import BPETokenizer, GPTLanguageModel, best_of_n, keyword_hits, rhyme_rate

ROOT = Path(__file__).resolve().parent
SITE_DIR = ROOT / "site"
CHECKPOINTS = [ROOT / "checkpoints" / "gpt_comptines_v2.pt", ROOT / "checkpoints" / "gpt_comptines_final.pt"]
# En local : 127.0.0.1:8000. Sur Hugging Face Spaces, le Dockerfile fixe HOST=0.0.0.0 et PORT=7860.
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 8000))
MAX_WORDS = 3

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
    return {
        "lines": text.split("\n"),
        "words": words,
        "found": [w for w in words if keyword_hits([w], text)],
        "rhyme_rate": rhyme_rate(text),
        "seconds": round(time.perf_counter() - start, 2),
    }


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
    print(f"site   : http://localhost:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
