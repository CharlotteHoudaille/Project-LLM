# Plan du projet — génération de texte à partir de mots-clés

Objectif : un petit GPT, entraîné en local, qui écrit un court texte anglais
pour enfants (histoire ou comptine) à partir de mots donnés.
`cat, moon` → un texte qui parle d'un chat et de la lune.

## Ce qui est déjà fait

| Notebook | Contenu | Résultat |
|---|---|---|
| `01_corpus` | 12 recueils de comptines Gutenberg, nettoyés et découpés | 2 981 comptines (408 000 caractères), 90 / 10 train / val |
| `02_entrainement` | GPT **caractère** (841 k paramètres), consigne `words: …` | val 1,49 ; 17 % des mots-clés repris (1,9 % au hasard) |
| `03_experiences` | Même modèle : 3 000 vs 10 000 pas, petit vs gros | point de comparaison caractère |

Constat : le modèle caractère apprend la **forme**, pas le **sens**, et
reprend mal les mots-clés. Deux causes : trop peu de données, et des mots à
épeler lettre par lettre.

## Choix validés pour la suite

| Question | Choix |
|---|---|
| Données | **Pré-entraînement sur TinyStories**, puis **fine-tuning sur les comptines** |
| Tokenizer | **tiktoken GPT-2**, réduit aux tokens présents dans nos données |
| Modèle | GPT du cours, **taille moyenne (~10 M paramètres)**, environ 1 h d'entraînement |
| Mots-clés | **Extraits automatiquement** (même règle que pour les comptines) |

## Les étapes

### Étape 1 — `04_donnees_bpe` : les données en tokens

- **TinyStories** : les ~50 premiers Mo de `TinyStoriesV2-GPT4-train.txt`
  (histoires écrites par GPT-4), téléchargés par requête HTTP partielle ;
  validation sur un extrait de `TinyStoriesV2-GPT4-valid.txt`. Les histoires
  sont séparées par `<|endoftext|>` dans le fichier.
- **Comptines** : les fichiers train / val du notebook 01, inchangés.
- **Format d'un exemple** :

  ```
  words: cat, moon, jump
  Once upon a time, a little cat ...<|endoftext|>
  ```

  `<|endoftext|>` (le token spécial de GPT-2) remplace le `\n\n` de fin.
- **Tokenisation** : `tiktoken.get_encoding("gpt2")`, comme au notebook 08.
- **Vocabulaire réduit** : on ne garde que les tokens GPT-2 vus dans nos
  données (attendu : 15 à 20 000 au lieu de 50 257), avec une table
  GPT-2 ↔ indice compact. Les 256 tokens d'un octet sont toujours gardés :
  un mot inconnu donné par l'utilisateur peut toujours s'écrire en morceaux.
- À mesurer : nombre d'histoires, de tokens, taille du vocabulaire,
  longueur des exemples en tokens (pour vérifier que `block_size = 256` suffit).

### Étape 2 — `05_preentrainement` : apprendre l'anglais et la consigne

- **Architecture** : le GPT du script 06 (blocs pre-norm, attention
  multi-têtes masquée, feed-forward, résiduels, dropout), avec :
  - `n_embd = 256`, `n_head = 8`, `n_layer = 6`, `block_size = 256` tokens,
    `dropout = 0.1` ;
  - **partage des poids** entre l'embedding et la couche de sortie (comme GPT-2).
- **Données d'entraînement** : un exemple = une histoire ; mots-clés (1 à 3)
  **retirés à chaque passage** ; loss **masquée** sur la consigne et le
  remplissage (`ignore_index=-1`).
- **Boucle** : celle du cours (pas, `RandomSampler`, TensorBoard, AdamW),
  avec le meilleur modèle de validation gardé de côté.
- **Budget** : nombre de pas fixé après une mesure de vitesse, pour environ 1 h.

**Résultat (fait)** : 99 minutes, validation 1,58 (perplexité 4,8). Le
modèle écrit des **histoires cohérentes**, mais il reprend mal les mots-clés :
42,7 % contre 21,7 % pour la référence. Beaucoup de mots-clés tirés au
hasard sont génériques (`time`, `day`, `happy`) et apparaissent dans presque
toutes les histoires : le modèle n'a pas besoin de lire la consigne pour les
« reprendre ».

Il écrit des histoires et pas des comptines : c'est normal à ce stade, il
n'a vu que TinyStories. La forme comptine vient à l'étape 4.

### Étape 3 — `06_mots_cles` : renforcer l'usage des mots-clés

- On **reprend le modèle pré-entraîné** et on continue ~3 000 pas sur
  TinyStories, avec un nouveau tirage des mots-clés : **pondéré par la
  rareté** du mot (IDF, *inverse document frequency*). `moon`, `pie`,
  `garden` sortent bien plus souvent que `day` ou `happy`. Pour bien prédire
  ces mots, le modèle doit lire la consigne.
- Comparaison avant / après, sur les mêmes consignes fixes : tirage uniforme
  (celui du notebook 05) et tirage pondéré.

### Étape 4 — `07_finetuning_comptines` : des comptines de 4 vers, qui riment un peu

- **Données** : uniquement les comptines de **4 lignes** (1 767 sur 2 981),
  même format `words: …` + comptine + `<|endoftext|>`, mots-clés pondérés.
- **Fine-tuning** : on reprend le modèle de l'étape 3, learning rate plus petit
  (~1e-4), peu de pas, meilleur modèle de validation gardé.
- **Génération** :
  - arrêt après la **4ᵉ ligne** (ou au `<|endoftext|>`) ;
  - **meilleur de N** : on génère plusieurs comptines pour la même consigne
    et on garde celle qui a le meilleur score (mots-clés repris + rimes).
    Le modèle ne change pas, seul le choix du texte change.
- **Mesure des rimes** : deux vers riment si leurs derniers mots sont
  différents et ont la même fin (dernière voyelle et ce qui suit, en tenant
  compte du *e* muet). Repère : 38,5 % des vers riment dans les comptines de
  4 vers d'entraînement, 5,2 % des lignes dans les histoires TinyStories.

**Résultat (fait)** — sur les 174 comptines de 4 vers de validation :

| modèle | bits / caractère | mots repris (réf.) | vers qui riment | 4 vers | mots par vers |
|---|---|---|---|---|---|
| 02 caractère | 2,15 | 15,0 % (0,0 %) | 5,4 % | 70,1 % | 7,9 |
| 05 pré-entraîné | 3,35 | 14,4 % (2,0 %) | 2,3 % | 29,9 % | 40,3 |
| 06 mots-clés | 3,40 | 15,6 % (3,2 %) | 2,6 % | 32,8 % | 40,9 |
| 07 fine-tuné | **1,50** | 7,4 % (1,8 %) | 7,1 % | 87,4 % | 6,3 |
| 07 + meilleur de 8 | | **34,2 %** (0,6 %) | **23,9 %** | **100 %** | 6,5 |
| vraies comptines | | | 41,2 % | 100 % | 6,5 |

La forme est acquise (vers courts, 4 vers), mais le fine-tuning a fait
**oublier une partie de l'usage des mots-clés** (7,4 % sans meilleur de N,
contre 15,6 % avant). Le meilleur de 8 compense.

### Étape 4 bis — `08_finetuning_melange` : garder les mots-clés

- Fine-tuning **mélangé** (1 histoire TinyStories pour 2 comptines), modèle
  choisi sur `mots repris + 0,5 × rimes` (pas 1 250 sur 1 500).
- **Coup de pouce** sur les mots-clés pas encore écrits (bonus 6 sur les
  logits, suivi token par token), combiné au **meilleur de 8**.
- **Test de copie** : part des vers générés présents tels quels dans le train.

Une première version (même consigne `words: …` pour les deux genres) reprenait
79 % des mots-clés, mais écrivait **souvent des histoires** (16 mots par
ligne) : le modèle ne pouvait pas savoir quel genre on voulait. Corrections :

- **une consigne par genre** (*control code*) : `rhyme: …` pour les comptines,
  `words: …` pour les histoires ; on génère avec `rhyme:` ;
- **une pénalité sur les lignes de plus de 12 mots** dans le meilleur de N
  (0,1 % des vrais vers dépassent 12 mots).

**Résultat (fait)** — 174 comptines de validation, modèle gardé au pas 500 :

| | bits / car. | mots repris (réf.) | tous les mots | vers qui riment | 4 vers | mots par vers | vers copiés |
|---|---|---|---|---|---|---|---|
| 07 seul | 1,55 * | 7,4 % (1,8 %) | 3,4 % | 7,1 % | 87,4 % | 6,3 | 26,8 % |
| 08 seul | 1,75 | 20,7 % (1,2 %) | 11,5 % | 2,7 % | 60,9 % | 6,6 | 1,3 % |
| 08 + coup de pouce | | 60,5 % (1,1 %) | 38,5 % | 3,9 % | 63,2 % | 6,3 | 0,7 % |
| 08 + meilleur de 8 | | 39,8 % (0,9 %) | 23,0 % | 7,3 % | 96,0 % | 6,2 | 0,9 % |
| 08 + les deux | | **80,9 %** (1,2 %) | **66,1 %** | 8,6 % | **98,9 %** | **5,9** | 1,0 % |
| vraies comptines | | | | 41,2 % | 100 % | 6,5 | 33,9 % |

\* mesuré ici avec la consigne `rhyme:`, que le modèle 07 n'a jamais vue
(1,50 au notebook 07, avec `words:`).

Le modèle écrit maintenant des **comptines de 4 vers courts** qui contiennent
tous les mots demandés dans deux cas sur trois, et il invente ses vers (1 %
de vers copiés). Reste faible : les **rimes** (8,6 % contre 41,2 %).
- **Comparaison finale** sur la validation des comptines (4 lignes) :
  - modèle caractère (notebook 02) / pré-entraîné / mots-clés renforcés / fine-tuné ;
  - loss en **bits par caractère** (une loss par token et une loss par
    caractère ne se comparent pas directement) ;
  - taux de reprise des mots-clés (et sa référence), taux de vers qui riment,
    nombre de lignes ;
  - les consignes de démonstration (`cat, moon`, `king, pie`, …).

### Étape 5 — `generer.py` : utiliser le modèle

```bash
uv run python generer.py cat moon --temperature 0.8 --top-k 40
```

Charge le checkpoint final et écrit une comptine de 4 vers à partir des mots
donnés (meilleur de N).

## Ce qu'on attend, et ce qu'on n'attend pas

- Attendu : une comptine de **4 vers courts**, en anglais simple et
  cohérent, qui reprend nettement plus souvent les mots-clés que le modèle
  caractère, avec **un peu de rime**.
- Pas attendu : des rimes à chaque vers, un sens parfait. C'est accepté :
  l'objectif est la génération de texte conditionnée.

## Limites connues, à mentionner dans le rendu

- TinyStories est **synthétique** (généré par GPT-3.5 / GPT-4), licence CDLA-Sharing-1.0.
- Les comptines ont été dédoublonnées sur le texte exact : des variantes
  proches peuvent se trouver à la fois en train et en validation.
- Un mot-clé hors du vocabulaire de TinyStories et des comptines sera recopié
  au mieux, mais le modèle ne saura pas quoi en faire.
