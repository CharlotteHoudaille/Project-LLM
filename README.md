# The Little Rhyme Maker — un petit GPT qui écrit des comptines

Projet du cours d'IA (ESILV, A5, AICG) : un modèle de génération de texte,
entraîné **en local**, qui reçoit 2 ou 3 mots anglais et écrit une
**comptine de 4 vers** qui les utilise.

```
words : cat, moon

Hey cat, where are you going?
I'm going to the moon
...
```

Tout est construit avec les outils du cours (séance 2, du bigramme au GPT) :
PyTorch, tiktoken, scikit-learn, matplotlib, TensorBoard, et un environnement
géré par [uv](https://docs.astral.sh/uv/).

## Essayer en 3 commandes

Le modèle final (`checkpoints/gpt_comptines_v2.pt`, 44 Mo) est inclus : pas besoin
de réentraîner pour l'essayer.

```bash
uv sync
uv run python serveur.py
```

Puis ouvrir **http://localhost:8000** : on choisit 2 ou 3 mots (en les tapant ou
en cliquant sur les bulles d'idées) et on clique sur *Generate a comptine*.

Sans le site, en ligne de commande :

```bash
uv run python generer.py cat moon
uv run python generer.py king pie --candidates      # montre les 8 comptines candidates
uv run python generer.py moon cat dog --boost 0 --n 1   # le modèle seul, sans aide
```

## Installation

1. **Installer uv** (il installe Python 3.13 tout seul) :
   ```bash
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
   curl -LsSf https://astral.sh/uv/install.sh | sh                                      # macOS / Linux
   ```
2. **Créer l'environnement** depuis la racine du dépôt : `uv sync`.
3. **GPU NVIDIA (optionnel, fortement conseillé pour réentraîner)** : sous Windows et
   Linux, le projet installe PyTorch pour CUDA 12.6. Il faut un pilote NVIDIA récent
   (version 525 ou plus). Vérifier :
   ```bash
   uv run python -c "import torch; print(torch.cuda.is_available())"
   ```
   Sans GPU, tout marche sur le CPU, mais les entraînements sont 10 à 15 fois plus longs.
   Sur Mac, PyTorch utilise la puce Apple (`mps`).

> Si le dossier est dans **OneDrive**, garder `link-mode = "copy"` dans
> `pyproject.toml` : OneDrive casse les liens physiques que uv utilise sinon.

## Comment le modèle a été construit

Les notebooks se lisent dans l'ordre ; chacun explique ce qu'il fait et pourquoi.
Durées mesurées sur une RTX 3050 Laptop (4 Go).

| Notebook | Étape | Durée |
| --- | --- | --- |
| `01_corpus` | 2 981 comptines du Project Gutenberg : nettoyage, découpage, mots-clés | ≈ 1 min |
| `02_entrainement` | Premier GPT **caractère** (0,8 M de paramètres) : la forme, pas le sens | ≈ 8 min |
| `03_experiences` | Comparaison de tailles de modèles caractère (facultatif, jamais allé au bout) | ≈ 40 min |
| `04_donnees_bpe` | Extrait de **TinyStories** + tokenisation **BPE** avec tiktoken, vocabulaire réduit à 14 122 tokens | ≈ 2 min |
| `05_preentrainement` | **Pré-entraînement** d'un GPT de 8,4 M de paramètres sur TinyStories | 1 h 40 |
| `06_mots_cles` | Mots-clés tirés selon leur rareté (IDF) pour que le modèle lise la consigne | 30 min |
| `07_finetuning_comptines` | **Fine-tuning** sur les comptines de 4 vers, meilleur de N, mesure des rimes | 20 min |
| `08_finetuning_melange` | Fine-tuning **mélangé** (comptines + histoires), consigne `rhyme:`, coup de pouce sur les mots-clés | 20 min |

Pour tout refaire : exécuter les notebooks 01, 02, 04, 05, 06, 07 puis 08, dans cet
ordre (chacun relit les modèles des précédents ; le 03 est facultatif). `PLAN.md`
détaille chaque choix et ses résultats.

### Le modèle

Un **GPT décodeur seul**, celui du script 06 du cours, en plus grand :
embeddings de token et de position, 6 blocs Transformer *pre-norm* (attention
multi-têtes masquée à 8 têtes, feed-forward, connexions résiduelles, LayerNorm),
`n_embd = 256`, fenêtre de 256 tokens, dropout. Ajouts par rapport au cours :
partage des poids entre embedding et sortie (comme GPT-2), *learning rate* avec
échauffement et décroissance en cosinus, calcul en `bfloat16`.

### La consigne

Chaque exemple d'entraînement commence par une ligne de mots-clés tirés du texte
lui-même, puis le texte, puis le token de fin :

```
rhyme: cat, moon
<la comptine>
<|endoftext|>
```

La loss ne porte que sur le texte, pas sur la consigne. Les comptines commencent
par `rhyme:` et les histoires TinyStories par `words:` : ce premier mot dit au
modèle quel genre écrire (*control code*).

### La génération

- arrêt après le 4ᵉ vers ;
- **coup de pouce** : un bonus sur les tokens des mots-clés pas encore écrits ;
- **meilleur de 8** : 8 comptines pour la même consigne, on garde celle qui reprend
  le plus de mots-clés, rime le mieux et n'a pas de vers trop longs.

Le modèle écrit toujours chaque comptine lui-même, token par token.

## Résultats

Sur les 174 comptines de 4 vers de validation :

| | Mots-clés repris | Tous les mots | Vers qui riment | 4 vers | Mots par vers | Vers copiés du corpus |
| --- | --- | --- | --- | --- | --- | --- |
| Modèle seul (08) | 20,7 % | 11,5 % | 2,7 % | 60,9 % | 6,6 | 1,3 % |
| + coup de pouce + meilleur de 8 | **80,9 %** | **66,1 %** | 8,6 % | 98,9 % | 5,9 | 1,0 % |
| Vraies comptines | — | — | 41,2 % | 100 % | 6,5 | 33,9 % |

**Ce qui marche** : la forme (4 vers courts), les mots-clés, et le modèle invente
ses vers au lieu de réciter le corpus.
**Ce qui reste faible** : le sens (8,4 M de paramètres, c'est minuscule) et les rimes.

## Structure du dépôt

```
.
├── 01_corpus.ipynb … 08_finetuning_melange.ipynb   les étapes, dans l'ordre
├── comptines.py        code partagé : tokenizers, datasets, GPT, génération, rimes
├── generer.py          génération en ligne de commande
├── serveur.py          le site, en local (bibliothèque standard uniquement)
├── site/               la page : index.html, style.css, app.js
├── checkpoints/        le modèle final (gpt_comptines_v2.pt)
├── data/               les comptines nettoyées, le vocabulaire et la table d'IDF
├── PLAN.md             les choix faits et les résultats, étape par étape
└── pyproject.toml, uv.lock
```

Les données téléchargées (`data/raw/`), les données en tokens (`data/bpe/*.pt`),
les autres modèles et les courbes TensorBoard (`runs/`) ne sont pas dans le dépôt :
les notebooks les recréent.

## Sources et licences

- Comptines : recueils du [Project Gutenberg](https://www.gutenberg.org/), domaine public.
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan et Li, 2023),
  histoires générées par GPT-3.5 et GPT-4, licence CDLA-Sharing-1.0.
- Le GPT suit la série *Neural Networks: Zero to Hero* d'Andrej Karpathy, reprise dans le cours.
- Polices du site : Baloo 2, Walter Turncoat et Plus Jakarta Sans (Google Fonts).
