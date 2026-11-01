# Semantic Liquidity NLP
### Laboratoire expérimental de calcul sémantique — Python × R × Zig × SLP

[![CI générale](https://github.com/tlpp7/semantic-liquidity-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/tlpp7/semantic-liquidity-nlp/actions/workflows/ci.yml)
[![Circuit R–Zig–Python](https://github.com/tlpp7/semantic-liquidity-nlp/actions/workflows/slp-circuit.yml/badge.svg)](https://github.com/tlpp7/semantic-liquidity-nlp/actions/workflows/slp-circuit.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Zig](https://img.shields.io/badge/Zig-C%20ABI%20prototype-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **Objet :** étudier des opérateurs sémantiques différentiables, comparer leurs comportements mathématiques, entraîner des modèles NLP et expérimenter un circuit de calcul natif interopérable R ↔ Zig ↔ Python. **Projet de recherche et d'ingénierie, non qualifié pour la production.**

**Navigation :** [Vue d'ensemble](#vue-densemble) · [Architecture](#architecture-en-trois-niveaux) · [Circuit SLP](#protocole-slp-v1) · [Mathématiques](#modèle-mathématique) · [Installation](#démarrage-rapide) · [Tests](#validation-et-ci) · [Roadmap](#feuille-de-route-p10p45) · [Limites](#limites-et-sécurité)

---

## Vue d'ensemble

Le dépôt contient **30 variantes d'opérateurs PyTorch** réparties en six familles, un classifieur de texte supervisé (embedding + GRU bidirectionnel), des outils d'entraînement et de benchmarking, une mémoire de replay, des expérimentations de couplage et de graphes de prototypes, ainsi qu'un **prototype de noyaux Zig** (Softmax et entropie) utilisables depuis Python et R.

| Domaine | Composant existant | Statut / portée |
|---|---|---|
| NLP | 30 opérateurs, entraînement, métriques et checkpoints | Prototype fonctionnel |
| Recherche | Benchmarks, validation, Optuna, replay/EWC | Expérimental |
| Graphes | Couplage P9 et prototypes P10 | PyTorch, expérimental |
| Calcul natif | Zig Softmax / entropie, ABI C, pont Python/R | Prototype CPU |
| Protocole | SLP v1 : encodage/décodage de matrices | **Framing binaire**, pas de réseau |
| Visualisation | Tableau de bord Streamlit | Disponible |
| Web D3.js / Three.js, DAG, agents | P28–P39 | **Planifié**, pas livré |
| Transport TCP/TLS et orchestration distribuée | P22–P27 | **Planifié**, pas livré |

## Architecture en trois niveaux

```mermaid
flowchart TB
  subgraph HIGH["NIVEAU HAUT — Applications et recherche"]
    direction LR
    PY["Python / PyTorch<br/>NLP, entraînement, gradients"]
    R["R<br/>Statistiques et vérification"]
    DASH["Streamlit<br/>Dashboard expérimental"]
  end
  subgraph MID["NIVEAU INTERMÉDIAIRE — Interface native"]
    direction LR
    ABI["ABI C / adaptateurs"]
    ZIG["Zig<br/>Softmax, entropie, validation"]
    FRAME["SLP v1<br/>Framing binaire"]
  end
  subgraph LOW["NIVEAU BAS — Exécution"]
    direction LR
    CPU["CPU"]
    RAM["Mémoire"]
    CLK["Horloge / mesures"]
  end
  DASH --> PY
  PY --> ABI
  R --> ABI
  PY -.-> FRAME
  R -.-> FRAME
  ABI --> ZIG
  ZIG --> CPU
  ZIG --> RAM
  CLK -.-> ZIG
```

**Attention :** R est un langage de haut niveau ; il peut participer aux calculs natifs en appelant l'ABI Zig. L'horloge mesure et synchronise, mais n'effectue pas elle-même les opérations arithmétiques.

### Circuit d'exécution réellement couvert par les tests

```mermaid
sequenceDiagram
    participant R as R / .C
    participant P as Python / ctypes
    participant S as SLP v1 (framing)
    participant Z as Bibliothèque Zig
    participant T as Tests / CI
    P->>S: encode_frame(matrix, request_id)
    S-->>P: decode_frame(bytes)
    P->>Z: sl_softmax_f64 / sl_entropy_f64
    Z-->>P: Matrices / statut
    R->>Z: sl_softmax_r / sl_entropy_r
    Z-->>R: Matrices / statut
    P->>T: Parité numérique / corruption
    R->>T: Parité numérique / layout
    Note over S,Z: Pas de transport réseau SLP implémenté
```

**À distinguer :** le round-trip SLP et les appels natifs Zig sont testés ; cela ne signifie pas qu'une requête SLP traverse déjà un serveur TCP ou un bus IPC.

## Protocole SLP v1

**Semantic Liquidity Protocol (SLP)** : format binaire expérimental de trames de matrices.

```mermaid
flowchart LR
  A["Header 32 octets"] --> B["Payload Float64<br/>little-endian"]
  B --> C["Matrice row-major"]
  A --- D["magic SLP1 · version<br/>op · dtype · flags<br/>rows · cols<br/>payload_bytes · request_id"]
```

| Champ | Encodage |
|---|---|
| Magic | `SLP1` |
| Version | `1` |
| Opération | `1` Softmax ; `2` entropie |
| Dtype | `1` = Float64 |
| Dimensions | `rows: u32`, `cols: u32` |
| Taille de charge | `payload_bytes: u64` |
| Corrélation | `request_id: u64` |
| Ordre des octets | Little-endian ; matrices row-major |
| Taille maximale | 128 MiB de payload |

La trame n'est **ni chiffrée ni authentifiée**. Ne pas l'exposer directement à Internet. Le transport réseau, la négociation de versions et les contrôles de sécurité sont des objectifs de roadmap.

## Modèle mathématique

Pour une représentation `X` et des prototypes `M` de dimension `d` :

```math
S = \frac{XM^\top}{\sqrt d}
```

```math
P_{ij}=\frac{\exp(S_{ij}-\max_k S_{ik})}{\sum_k\exp(S_{ik}-\max_k S_{ik})}
```

```math
H_i=-\frac{\sum_j P_{ij}\log(P_{ij})}{\log K}
```

Invariants numériques : `P >= 0`, somme des probabilités par ligne proche de `1`, entropie normalisée dans `[0, 1]` sous hypothèses valides. La bibliothèque Zig n'implémente pas automatiquement la rétropropagation PyTorch.

### Cycle expérimental

```mermaid
flowchart TD
  A["Jeu de données étiqueté"] --> B["Catalogue et provenance P5"]
  B --> C["Partition train / validation / test P4"]
  C --> D["Entraînement P1"]
  D --> E["Opérateurs baseline / 30 variantes / P9–P10"]
  E --> F["Métriques et télémétrie P6"]
  F --> G["Benchmarks P2 / tuning P7"]
  G --> H{"Évaluation reproductible ?"}
  H -->|Non| I["Corriger protocole et contraintes"]
  I --> C
  H -->|Oui| J["Rapport expérimental"]
```

## Démarrage rapide

**Pré-requis :** Python ≥ 3.10. Les fonctionnalités Zig et R nécessitent leurs toolchains respectives.

```bash
git clone https://github.com/tlpp7/semantic-liquidity-nlp.git
cd semantic-liquidity-nlp
python -m pip install -e ".[dev,lab]"
python -m pytest -q
streamlit run dashboard/app.py
```

Pour installer uniquement les dépendances de test : `python -m pip install -e ".[dev]"`.

### Préparer les données

Le dashboard accepte un CSV étiqueté (`text`, `target`). Le catalogue permet d'enregistrer source, licence déclarée, classes et empreinte SHA-256 :

```bash
python scripts/catalog.py csv my_dataset.csv --text-col text --label-col target --license CC0
python scripts/catalog.py list
```

Téléchargement Kaggle optionnel : `python scripts/kaggle_download.py owner/dataset-name`. Authentification et droits de réutilisation restent à vérifier.

### Entraîner et comparer

```bash
python scripts/train.py --dataset storage/dataset.jsonl --operator baseline --epochs 5 --seed 42
python scripts/train.py --dataset storage/dataset.jsonl --operator coupled_v2 --epochs 5 --seed 42
python scripts/train.py --dataset storage/dataset.jsonl --operator graph_semantic --epochs 5 --seed 42

python scripts/benchmark.py --dataset storage/dataset.jsonl --operators baseline coupled_v2 graph_semantic --seeds 42 43 44 --epochs 5
```

Autres expériences :

```bash
python scripts/tune.py --dataset storage/dataset.jsonl --operators baseline softmax_beta coupled_dynamics --trials 12 --epochs 8
python scripts/continual.py --dataset storage/dataset.jsonl --tasks 3 --epochs 2 --ewc-lambda 1.0 --replay-capacity 256 --replay-samples 64
```

Les expériences enregistrent leurs sorties localement sous `storage/`. Les métriques, la stabilité et la généralisation doivent être évaluées sur des données appropriées.

## Validation et CI

La CI générale lance **deux branches de vérification**, puis une porte finale :

```mermaid
flowchart TD
  A["Push / Pull Request"] --> P["Suite Python complète<br/>pytest"]
  A --> N["Compilation Zig 0.13"]
  N --> Z["zig test SLP"]
  Z --> NP["Python ↔ Zig : ABI et parité"]
  NP --> NR["R ↔ Zig : ABI et parité"]
  P --> G{"CI Gate"}
  NR --> G
  G -->|Deux jobs réussis| OK["Validation CI"]
  G -->|Un job échoue| FAIL["Échec : correction requise"]
```

- [CI générale](https://github.com/tlpp7/semantic-liquidity-nlp/actions/workflows/ci.yml)
- [Tests spécialisés SLP R–Zig–Python](https://github.com/tlpp7/semantic-liquidity-nlp/actions/workflows/slp-circuit.yml)
- [Issues de développement](https://github.com/tlpp7/semantic-liquidity-nlp/issues)

**Exécution manuelle du noyau natif (Zig 0.13) :**

```bash
zig build-lib -dynamic -O ReleaseSafe native/zig/combined.zig -femit-bin=libsemantic_core.so
zig test native/zig/slp_protocol.zig
SLP_ZIG_LIBRARY="$PWD/libsemantic_core.so" python -m pytest -q tests/test_slp_circuit.py
SLP_ZIG_LIBRARY="$PWD/libsemantic_core.so" Rscript tests/test_slp_r_circuit.R
```

Commandes adaptées à Linux ; sous Windows/macOS, ajuster le nom de la bibliothèque partagée. Le workflow GitHub Actions constitue la référence automatisée.

## Feuille de route P10–P45

**Les six blocs suivants sont documentés et suivis par des issues ; ils ne sont pas tous implémentés.** Le P10 historique (graphe sémantique PyTorch) est distinct du P10 de la feuille de route d'industrialisation (audit du socle).

```mermaid
gantt
  title Roadmap logique — séquence indicative, sans dates promises
  dateFormat X
  axisFormat %s
  section Fondations
  Bloc 1 — P10 à P15 : b1, 0, 6
  section Noyaux
  Bloc 2 — P16 à P21 : b2, after b1, 6
  section Distribution
  Bloc 3 — P22 à P27 : b3, after b2, 6
  section Intelligence
  Bloc 4 — P28 à P33 : b4, after b3, 6
  section Interfaces
  Bloc 5 — P34 à P39 : b5, after b4, 6
  section Qualification
  Bloc 6 — P40 à P45 : b6, after b5, 6
```

| Bloc | Phases | Objectif | Détail |
|---|---|---|---|
| 1 | P10–P15 | Audit, contrat d'opérateurs, SLP v2, IPC et dispatch | [Roadmap 1](docs/ROADMAP_BLOCK_1_P10_P15.md) |
| 2 | P16–P21 | Algèbre native, SIMD, graphes, gradients et benchmarks | [Roadmap 2](docs/ROADMAP_BLOCK_2_P16_P21.md) |
| 3 | P22–P27 | TCP/TLS, nœuds distribués, reprise et sécurité | [Roadmap 3](docs/ROADMAP_BLOCK_3_P22_P27.md) |
| 4 | P28–P33 | DAG, recherche d'architectures, mémoire, MPC, agents | [Roadmap 4](docs/ROADMAP_BLOCK_4_P28_P33.md) |
| 5 | P34–P39 | API, interface web, D3.js, Three.js, éditeur DAG | [Roadmap 5](docs/ROADMAP_BLOCK_5_P34_P39.md) |
| 6 | P40–P45 | Validation scientifique, CI/CD, déploiement et v1.0 | [Roadmap 6](docs/ROADMAP_BLOCK_6_P40_P45.md) |

### Vue des dépendances

```mermaid
flowchart LR
  B1["B1 · Fondations"] --> B2["B2 · Calcul"]
  B1 --> B3["B3 · Distribution"]
  B2 --> B4["B4 · Intelligence"]
  B3 --> B4
  B3 --> B5["B5 · Interfaces"]
  B4 --> B5
  B2 --> B6["B6 · Qualification"]
  B5 --> B6
```

Les **36 issues ouvertes** correspondent aux tâches de la roadmap et ne sont pas 36 bugs identifiés.

## Structure du dépôt

```text
src/semantic_liquidity/   # Opérateurs, NLP, mémoire, entraînement, validation
native/zig/               # Noyaux CPU, ABI R, SLP
r/                        # Référence mathématique et pont natif
scripts/                  # CLI train, benchmark, tuning, protocole, bridge
dashboard/                # Interface Streamlit
tests/                    # Tests Python et R
docs/                     # Architecture et six blocs de roadmap
.github/workflows/        # CI générale et circuit SLP
storage/                  # Données et artefacts locaux (ignorés par Git)
```

Documentation technique : [Architecture à trois niveaux](docs/THREE_LEVEL_ARCHITECTURE.md) · [Synthèse P0–P10](docs/PRESENTATION_P0_P10.md).

## Limites et sécurité

- Les opérateurs sémantiques, la plasticité et la consolidation sont des **proxies mathématiques** ; aucune équivalence biologique n'est établie.
- Les comparaisons de modèles ne prouvent pas de supériorité sans baselines contrôlées, graines multiples et tests indépendants.
- L'évaluation actuelle de l'apprentissage continu réutilise des données d'entraînement : **ce n'est pas un benchmark held-out**.
- La sélection top-k des voisins dans les graphes est seulement différentiable par morceaux.
- La parité des noyaux Zig/R/Python ne prouve ni la sûreté réseau ni l'existence d'un protocole distribué opérationnel.
- Ne jamais publier de tokens Kaggle, secrets, données privées, checkpoints ou traces contenant des informations sensibles.

**Licence :** MIT. Voir le fichier de licence du dépôt.
