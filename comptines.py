"""Code commun du projet, repris du notebook 02.

Le notebook 02 construit chaque morceau pas à pas ; les notebooks suivants
l'importent d'ici au lieu de le recopier.
"""

import random
import re

import tiktoken
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from torch import nn
from torch.utils.data import Dataset


# --- Mots-clés et consigne ---------------------------------------------------


def content_words(rhyme: str) -> list[str]:
    """Les mots porteurs de sens d'une comptine, sans doublon, dans l'ordre."""
    words = re.findall(r"[a-z]+", rhyme.lower())
    return list(dict.fromkeys(w for w in words if len(w) >= 3 and w not in ENGLISH_STOP_WORDS))


def sample_keywords(words: list[str], rng: random.Random, max_keywords: int = 3,
                    idf: dict[str, float] | None = None) -> list[str]:
    """Tire de 1 à `max_keywords` mots-clés parmi `words`.

    Sans `idf`, tous les mots ont la même chance. Avec `idf` (notebook 06), la
    chance d'un mot est proportionnelle à son IDF : les mots rares sortent
    plus souvent que les mots présents partout.
    """
    if not words:
        return []
    k = rng.randint(1, min(max_keywords, len(words)))
    if idf is None:
        return rng.sample(words, k=k)
    # Tirage pondéré sans remise (Efraimidis et Spirakis) : chaque mot reçoit
    # la clé u^(1/poids), u uniforme sur ]0, 1[ ; on garde les k plus grandes.
    default = max(idf.values())
    keys = {w: rng.random() ** (1 / idf.get(w, default)) for w in words}
    return sorted(words, key=keys.get, reverse=True)[:k]


def pick_keywords(rhyme: str, rng: random.Random, max_keywords: int = 3) -> list[str]:
    """Tire de 1 à `max_keywords` mots-clés dans la comptine."""
    return sample_keywords(content_words(rhyme), rng, max_keywords)


def make_prompt(keywords: list[str], genre: str = "words") -> str:
    """La consigne : la ligne des mots-clés, précédée du genre voulu.

    `words` est la consigne des notebooks 02 à 07 (et des histoires au
    notebook 08) ; `rhyme` celle des comptines au notebook 08. Ce premier mot
    dit au modèle quel genre de texte écrire (*control code*).
    """
    return f"{genre}: {', '.join(keywords)}\n"


def keyword_hits(keywords: list[str], text: str) -> float:
    """Part des mots-clés présents (comme mots entiers) dans le texte."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    return sum(k in words for k in keywords) / len(keywords)


# --- Rimes (notebook 07) -------------------------------------------------------


def rhyme_key(line: str) -> str | None:
    """La « fin qui rime » du dernier mot d'un vers : dernière voyelle et ce qui suit.

    `hill` → `ill`, `day` → `ay`, `moon` → `oon`. Un *e* muet final compte
    avec la voyelle d'avant : `time` → `ime`, `more` → `ore`.
    """
    words = re.findall(r"[a-z']+", line.lower())
    if not words:
        return None
    word = words[-1].strip("'")
    silent_e = len(word) > 2 and word.endswith("e") and word[-2] not in "aeiouy"
    stem = word[:-1] if silent_e else word
    match = re.search(r"[aeiouy]+[^aeiouy]*$", stem)
    return (match.group(0) if match else stem) + ("e" if silent_e else "")


def last_word(line: str) -> str | None:
    """Le dernier mot d'un vers, en minuscules."""
    words = re.findall(r"[a-z']+", line.lower())
    return words[-1].strip("'") if words else None


def rhyme_rate(text: str) -> float:
    """Part des vers qui riment avec le vers suivant ou celui d'après (AABB, ABAB).

    Deux vers riment si leurs derniers mots ont la même `rhyme_key` et sont
    différents : répéter le même mot (`moon` / `moon`) n'est pas une rime.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < 2:
        return 0.0
    keys = [rhyme_key(line) for line in lines]
    ends = [last_word(line) for line in lines]
    rhyming = set()
    for i in range(len(lines)):
        for j in (i + 1, i + 2):
            if j < len(lines) and keys[i] and keys[i] == keys[j] and ends[i] != ends[j]:
                rhyming |= {i, j}
    return len(rhyming) / len(lines)


# --- Vocabulaire caractère ---------------------------------------------------


class CharTokenizer:
    """Vocabulaire caractère du script 01 : `encode` / `decode`."""

    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        """Texte → liste d'indices."""
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        """Liste d'indices → texte."""
        return "".join(self.itos[i] for i in ids)


# --- Vocabulaire BPE (notebook 04) -------------------------------------------


class BPETokenizer:
    """Le découpage de tiktoken GPT-2, réduit aux tokens de `gpt2_ids`.

    Un token GPT-2 absent du vocabulaire réduit est écrit octet par octet :
    les 256 tokens d'un octet font toujours partie de `gpt2_ids`.

    Args:
        gpt2_ids: Les identifiants GPT-2 gardés ; leur position est l'indice compact.
    """

    def __init__(self, gpt2_ids: list[int]) -> None:
        self.enc = tiktoken.get_encoding("gpt2")
        self.gpt2_ids = gpt2_ids
        self.vocab_size = len(gpt2_ids)
        self.to_compact = {t: i for i, t in enumerate(gpt2_ids)}
        self.byte_ids = [self.to_compact[self.enc.encode_single_token(bytes([b]))] for b in range(256)]
        self.eot = self.to_compact[self.enc.eot_token]  # <|endoftext|>
        # Nombre de "\n" dans chaque token (notebook 07), calculé une fois pour toutes.
        self.newline_counts = torch.tensor([self.enc.decode_single_token_bytes(t).count(b"\n") for t in gpt2_ids])

    def encode(self, s: str) -> list[int]:
        """Texte → liste d'indices compacts."""
        ids = []
        for t in self.enc.encode_ordinary(s):
            if t in self.to_compact:
                ids.append(self.to_compact[t])
            else:  # token jamais vu : on l'écrit octet par octet
                ids.extend(self.byte_ids[b] for b in self.enc.decode_single_token_bytes(t))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Liste d'indices compacts → texte."""
        return self.enc.decode([self.gpt2_ids[i] for i in ids])


# --- Dataset -----------------------------------------------------------------


class RhymeDataset(Dataset):
    """Une comptine par exemple : consigne de mots-clés, comptine, ligne vide.

    Les cibles valent -1 sur la consigne et sur le remplissage : la loss les ignore.

    Args:
        rhymes: Les comptines.
        tokenizer: Le vocabulaire.
        block_size: Taille de la fenêtre ; les exemples plus longs sont coupés.
        fixed_keywords: Si vrai, les mots-clés de la comptine `i` sont
            toujours les mêmes (validation) ; sinon, ils changent à chaque appel.
    """

    def __init__(self, rhymes: list[str], tokenizer: CharTokenizer, block_size: int,
                 fixed_keywords: bool = False) -> None:
        self.rhymes = rhymes
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.fixed_keywords = fixed_keywords

    def __len__(self) -> int:
        return len(self.rhymes)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        rhyme = self.rhymes[i]
        rng = random.Random(i) if self.fixed_keywords else random
        prompt = make_prompt(pick_keywords(rhyme, rng))
        ids = torch.tensor(self.tokenizer.encode(prompt + rhyme + "\n\n")[: self.block_size + 1])
        n = len(ids) - 1

        x = torch.full((self.block_size,), self.tokenizer.stoi["\n"], dtype=torch.long)  # remplissage
        y = torch.full((self.block_size,), -1, dtype=torch.long)  # -1 : position ignorée par la loss
        x[:n], y[:n] = ids[:-1], ids[1:]
        y[: len(prompt) - 1] = -1  # on ne note pas la consigne
        return x, y


class TokenTextDataset(Dataset):
    """Version BPE de `RhymeDataset`, sur un corpus sauvegardé par le notebook 04.

    Un exemple : consigne de mots-clés, texte, `<|endoftext|>`. Cibles à -1 sur
    la consigne et sur le remplissage.

    Args:
        packed: Le corpus : `ids`, `offsets`, `keywords` (notebook 04).
        tokenizer: Le vocabulaire réduit.
        block_size: Taille de la fenêtre ; les exemples plus longs sont coupés.
        fixed_keywords: Si vrai, les mots-clés du texte `i` sont toujours les mêmes.
        idf: Si fourni, tirage des mots-clés pondéré par la rareté (notebook 06).
        genre: Le premier mot de la consigne (`words`, ou `rhyme` pour les comptines au notebook 08).
    """

    def __init__(self, packed: dict, tokenizer: BPETokenizer, block_size: int, fixed_keywords: bool = False,
                 idf: dict[str, float] | None = None, genre: str = "words") -> None:
        self.ids = packed["ids"].long()
        self.offsets = packed["offsets"]
        self.keywords = packed["keywords"]
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.fixed_keywords = fixed_keywords
        self.idf = idf
        self.genre = genre

    def __len__(self) -> int:
        return len(self.keywords)

    def text_ids(self, i: int) -> torch.Tensor:
        """Les tokens du texte `i`, sans consigne ni token de fin."""
        return self.ids[self.offsets[i] : self.offsets[i + 1]]

    def keywords_of(self, i: int) -> list[str]:
        """Les mots-clés du texte `i` : fixes ou tirés à nouveau selon `fixed_keywords`."""
        rng = random.Random(i) if self.fixed_keywords else random
        return sample_keywords(self.keywords[i], rng, idf=self.idf)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = torch.tensor(self.tokenizer.encode(make_prompt(self.keywords_of(i), self.genre)))
        eot = torch.tensor([self.tokenizer.eot])
        ids = torch.cat((prompt, self.text_ids(i), eot))[: self.block_size + 1]
        n = len(ids) - 1

        x = torch.full((self.block_size,), self.tokenizer.eot, dtype=torch.long)  # remplissage
        y = torch.full((self.block_size,), -1, dtype=torch.long)  # -1 : position ignorée par la loss
        x[:n], y[:n] = ids[:-1], ids[1:]
        y[: len(prompt) - 1] = -1  # on ne note pas la consigne
        return x, y


# --- Modèle : le GPT du script 06 ---------------------------------------------


class Head(nn.Module):
    """Une tête de self-attention masquée."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    """Plusieurs têtes en parallèle, concaténées puis projetées."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList(Head(n_embd, n_embd // n_head, block_size, dropout) for _ in range(n_head))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    """MLP appliqué à chaque position indépendamment."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Bloc Transformer pre-norm : attention puis feed-forward, avec résiduels."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GPTLanguageModel(nn.Module):
    """GPT du script 06 ; la loss ignore les cibles à -1.

    Args:
        tie_weights: Partage la matrice de l'embedding de token et celle de
            `lm_head`, comme GPT-2 (notebook 05). Les poids sont alors
            initialisés comme dans GPT-2 : loi normale d'écart-type 0,02.
    """

    def __init__(self, vocab_size: int, block_size: int, n_embd: int, n_head: int, n_layer: int,
                 dropout: float, tie_weights: bool = False) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        if tie_weights:
            self.apply(self._init_gpt2)
            self.lm_head.weight = self.token_embedding_table.weight  # une seule matrice (vocab_size, n_embd)

    @staticmethod
    def _init_gpt2(module: nn.Module) -> None:
        """Initialisation de GPT-2 : N(0, 0.02) pour les poids, 0 pour les biais."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        if targets is None:
            # Génération : seule la dernière position sert. Calculer les logits
            # partout coûterait (B, T, vocab_size) en mémoire pour rien.
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        return logits, F.cross_entropy(logits.view(B * T, -1), targets.view(B * T), ignore_index=-1)


# --- Génération ----------------------------------------------------------------


@torch.no_grad()
def write_rhyme(model: GPTLanguageModel, tokenizer: CharTokenizer, words: list[str],
                temperature: float = 0.8, top_k: int | None = 10, max_new_tokens: int = 400) -> str:
    """Écrit une comptine à partir de mots-clés (température et top-k du script 07).

    S'arrête dès que le modèle écrit une ligne vide.
    """
    model.eval()
    device = next(model.parameters()).device
    words = [re.sub(r"[^a-z]", "", w.lower()) for w in words]
    prompt = make_prompt([w for w in words if w])
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.block_size :])[0][:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)), dim=1)
        if tokenizer.decode(idx[0, -2:].tolist()) == "\n\n":  # ligne vide : la comptine est finie
            break
    return tokenizer.decode(idx[0, len(prompt) :].tolist()).strip()


@torch.no_grad()
def write_text(model: GPTLanguageModel, tokenizer: BPETokenizer, words: list[str],
               temperature: float = 0.8, top_k: int | None = 40, max_new_tokens: int = 300) -> str:
    """Version BPE de `write_rhyme` : s'arrête au token `<|endoftext|>`."""
    model.eval()
    device = next(model.parameters()).device
    words = [w.strip().lower() for w in words if w.strip()]
    prompt = tokenizer.encode(make_prompt(words))
    idx = torch.tensor([prompt], device=device)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.block_size :])[0][:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        if idx_next.item() == tokenizer.eot:  # le modèle a fini son texte
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return tokenizer.decode(idx[0, len(prompt) :].tolist()).strip()


@torch.no_grad()
def write_texts(model: GPTLanguageModel, tokenizer: BPETokenizer, keyword_sets: list[list[str]],
                temperature: float = 0.8, top_k: int | None = 40, max_new_tokens: int = 250,
                max_batch: int = 32, n_lines: int | None = None, keyword_boost: float = 0.0,
                genre: str = "words") -> list[str]:
    """`write_text` pour beaucoup de consignes à la fois, en lots.

    Les consignes de même longueur (en tokens) sont générées ensemble, par lots
    d'au plus `max_batch` : un lot est alors un tenseur rectangulaire, sans
    remplissage. Chaque texte est coupé à son premier `<|endoftext|>`.

    Avec `n_lines` (notebook 07), un texte s'arrête aussi dès sa `n_lines`-ième
    fin de ligne, et il est coupé à ses `n_lines` premières lignes.

    Avec `keyword_boost` (notebook 08), on ajoute ce bonus aux logits des tokens
    des mots-clés **pas encore écrits** dans le texte : ils deviennent plus
    probables, et le modèle choisit toujours où et comment les placer.

    `genre` est le premier mot de la consigne (`make_prompt`).
    """
    model.eval()
    device = next(model.parameters()).device
    if n_lines is not None:
        newlines = tokenizer.newline_counts.to(device)  # un token peut contenir 0, 1 ou plusieurs "\n"
    prompts = [tokenizer.encode(make_prompt(k, genre)) for k in keyword_sets]
    texts = [""] * len(prompts)
    by_length: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        by_length.setdefault(len(p), []).append(i)
    batches = [(length, indices[j : j + max_batch])
               for length, indices in by_length.items() for j in range(0, len(indices), max_batch)]

    for length, indices in batches:
        idx = torch.tensor([prompts[i] for i in indices], device=device)
        done = torch.zeros(len(indices), dtype=torch.bool, device=device)
        line_count = torch.zeros(len(indices), dtype=torch.long, device=device)
        if keyword_boost:
            trackers = [KeywordTracker(tokenizer, keyword_sets[i]) for i in indices]
        for _ in range(max_new_tokens):
            logits = model(idx[:, -model.block_size :])[0][:, -1, :] / temperature
            if keyword_boost:
                # Coup de pouce : + keyword_boost sur le prochain token des mots-clés pas encore écrits,
                # pour tout le lot en une seule opération.
                pairs = [(row, t) for row, tracker in enumerate(trackers) for t in tracker.next_tokens()]
                if pairs:
                    rows_idx, tokens_idx = zip(*pairs)
                    logits[list(rows_idx), list(tokens_idx)] += keyword_boost
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if keyword_boost:
                for tracker, token in zip(trackers, idx_next[:, 0].tolist()):
                    tracker.update(token)
            done |= idx_next[:, 0] == tokenizer.eot
            if n_lines is not None:
                line_count += newlines[idx_next[:, 0]]
                done |= line_count >= n_lines
            if done.all():
                break
        for row, i in enumerate(indices):
            generated = idx[row, length:].tolist()
            if tokenizer.eot in generated:
                generated = generated[: generated.index(tokenizer.eot)]
            text = tokenizer.decode(generated).strip()
            if n_lines is not None:
                text = "\n".join([line.strip() for line in text.split("\n") if line.strip()][:n_lines])
            texts[i] = text
    return texts


class KeywordTracker:
    """Suit, token par token, quels mots-clés d'une consigne ont déjà été écrits.

    Un mot s'écrit de quatre façons : avec ou sans espace devant (milieu ou
    début de vers), en minuscule ou avec une majuscule. Chaque forme est une
    suite de tokens (un seul pour 97 % des mots de TinyStories, plusieurs pour
    les mots rares). Pour chaque forme, on retient combien de ses tokens
    viennent d'être écrits à la suite.

    Deux garde-fous, pour ne pas faire dérailler le modèle :

    - on ne favorise que les formes d'au plus `max_tokens` tokens dont le
      premier token fait au moins 2 caractères. Un mot absent du vocabulaire
      s'écrit octet par octet (notebook 04) : favoriser une simple lettre
      ferait écrire n'importe quel mot qui commence par elle ;
    - après `max_failures` débuts de mot abandonnés, on cesse de le favoriser.
    """

    def __init__(self, tokenizer: BPETokenizer, keywords: list[str], max_tokens: int = 3,
                 max_failures: int = 3) -> None:
        self.forms = []
        for w in keywords:
            seqs = [tokenizer.encode(f) for f in {w, w.capitalize(), " " + w, " " + w.capitalize()}]
            self.forms.append([s for s in seqs if len(s) <= max_tokens and len(tokenizer.decode(s[:1]).strip()) >= 2])
        self.progress = [[0] * len(forms) for forms in self.forms]
        self.done = [not forms for forms in self.forms]  # aucune forme utilisable : pas de coup de pouce
        self.failures = [0] * len(keywords)
        self.max_failures = max_failures

    def next_tokens(self) -> list[int]:
        """Les tokens à favoriser : la suite de chaque forme des mots pas encore écrits.

        Si un mot est commencé, seule sa suite est favorisée ; sinon, son premier token.
        """
        boosted = []
        for k, forms in enumerate(self.forms):
            if self.done[k]:
                continue
            started = [seq[p] for seq, p in zip(forms, self.progress[k]) if p > 0]
            boosted += started or [seq[0] for seq in forms]
        return boosted

    def update(self, token: int) -> None:
        """Avance le suivi après l'écriture de `token`."""
        for k, forms in enumerate(self.forms):
            if self.done[k]:
                continue
            interrupted = False
            for f, seq in enumerate(forms):
                p = self.progress[k][f]
                if seq[p] == token:
                    p += 1
                else:  # la forme est interrompue : elle ne peut que recommencer
                    interrupted |= p > 0
                    p = 1 if seq[0] == token else 0
                if p == len(seq):
                    self.done[k], p = True, 0
                self.progress[k][f] = p
            if interrupted and not self.done[k]:
                self.failures[k] += 1
                self.done[k] = self.failures[k] >= self.max_failures  # on abandonne ce mot


def poem_score(keywords: list[str], text: str, n_lines: int = 4, keyword_weight: float = 1.0,
               max_words: int | None = None) -> float:
    """Score d'une comptine pour le meilleur de N : mots-clés repris (× `keyword_weight`),
    vers qui riment (× 0,5), pénalité de 1 si elle n'a pas exactement `n_lines` vers.

    Avec `max_words` (notebook 08), une seconde pénalité : la part des vers de plus
    de `max_words` mots. Une ligne de prose (une histoire) n'est pas un vers.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    format_penalty = 0.0 if len(lines) == n_lines else 1.0
    if max_words is not None and lines:
        format_penalty += sum(len(line.split()) > max_words for line in lines) / len(lines)
    return keyword_weight * keyword_hits(keywords, text) + 0.5 * rhyme_rate(text) - format_penalty


def best_of_n(model: GPTLanguageModel, tokenizer: BPETokenizer, words: list[str], n: int = 8,
              n_lines: int = 4, keyword_weight: float = 1.0, max_words: int | None = None,
              **kwargs) -> tuple[str, list[str]]:
    """Génère `n` comptines pour la même consigne et garde celle de meilleur `poem_score`.

    Returns:
        La meilleure comptine, et toutes les candidates.
    """
    words = [w.strip().lower() for w in words if w.strip()]
    candidates = write_texts(model, tokenizer, [words] * n, n_lines=n_lines, **kwargs)
    best = max(candidates, key=lambda text: poem_score(words, text, n_lines, keyword_weight, max_words))
    return best, candidates
