# PROJECT_CONTEXT.md
## Bug → Testcase Recommendation System
> Single source of truth for development context, architecture, and roadmap.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [System Architecture](#3-system-architecture)
4. [Development Phases](#4-development-phases)
5. [Coding Standards](#5-coding-standards)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Current Project Status](#7-current-project-status)

---

## 1. Project Overview

### Mission

The Bug → Testcase Recommendation System is an industrial ML pipeline that automatically predicts which test cases are most likely relevant to a newly reported bug. Given a new bug, the system surfaces a ranked Top-K list of test cases that engineers should execute — reducing manual triage time and improving test coverage.

### Prediction Goal

**Input:** A new bug (title, description, metadata such as component, priority, area path).

**Output:** An ordered list of Top-K test case IDs, ranked by predicted relevance.

### Motivation

Manual selection of test cases for each incoming bug is time-consuming and error-prone at scale. With ~33k test cases in the corpus, naive search is insufficient. This system leverages historical bug↔testcase linkage data to learn and generalize relevance patterns using a retrieval-augmented ranking architecture.

### Why Not Direct Classification?

Direct multiclass classification over ~33k test case labels is impractical:
- Extreme label sparsity (most test cases linked to very few bugs).
- New test cases would require model retraining.
- The problem is better framed as **retrieval + ranking**, which generalizes to unseen bugs and scales to new test cases without retraining.

---

## 2. Dataset Description

### 2.1 Source

All data is extracted from **Azure DevOps** via the REST API using Personal Access Token (PAT) authentication.

### 2.2 Entities

#### Bugs (`bugs.csv`)

Each row represents a historical bug work item.

| Field | Type | Description |
|---|---|---|
| `bug_id` | int | Unique Azure DevOps work item ID |
| `title` | str | Bug title (primary text signal) |
| `description` | str | Detailed bug description (HTML-stripped) |
| `area_path` | str | Organizational area (e.g., `Product\Backend`) |
| `iteration_path` | str | Sprint/iteration identifier |
| `priority` | int | Bug priority (1–4) |
| `severity` | str | Severity label (e.g., `2 - High`) |
| `state` | str | Lifecycle state (e.g., `Active`, `Resolved`, `Closed`) |
| `assigned_to` | str | Assignee display name |
| `created_date` | datetime | Bug creation timestamp |
| `resolved_date` | datetime | Resolution timestamp (nullable) |
| `tags` | str | Semicolon-separated tag string |

#### Test Cases (`testcases.csv`)

Each row represents a test case work item.

| Field | Type | Description |
|---|---|---|
| `testcase_id` | int | Unique Azure DevOps work item ID |
| `title` | str | Test case title |
| `steps` | str | Test steps (HTML-stripped) |
| `area_path` | str | Organizational area |
| `iteration_path` | str | Sprint/iteration identifier |
| `state` | str | Lifecycle state |
| `assigned_to` | str | Assignee display name |
| `priority` | int | Test case priority |
| `automated_test_name` | str | Automation identifier (nullable) |
| `tags` | str | Semicolon-separated tag string |

#### Historical Mappings (`master_dataset.csv`)

Each row represents a confirmed bug↔testcase link extracted from Azure DevOps test run / test plan relationships.

| Field | Type | Description |
|---|---|---|
| `bug_id` | int | Foreign key → `bugs.csv` |
| `testcase_id` | int | Foreign key → `testcases.csv` |
| `link_type` | str | Relationship type (e.g., `Tested By`) |
| `linked_date` | datetime | Date the link was established (nullable) |

### 2.3 Scale

| Entity | Approximate Count |
|---|---|
| Test Cases | ~33,000 |
| Historical Bug↔Testcase Mappings | ~5,000 |
| Unique Bugs with Mappings | Derived from mappings |

### 2.4 Key Characteristics

- A single bug may be linked to **multiple test cases**.
- A single test case may be linked to **multiple bugs** (shared coverage).
- The mapping distribution is **long-tailed** — a small number of test cases are heavily linked; most have sparse coverage.
- Text quality varies: some bugs have rich descriptions; others contain only a short title.

---

## 3. System Architecture

### 3.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                        NEW BUG INPUT                        │
│          (title + description + metadata)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               STAGE 1 — HISTORICAL BUG RETRIEVAL            │
│                                                             │
│  Embed new bug using intfloat/e5-large-v2                   │
│  Query FAISS index of historical bug embeddings             │
│  Retrieve Top-N most similar historical bugs                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            STAGE 2 — CANDIDATE TESTCASE AGGREGATION         │
│                                                             │
│  For each retrieved similar bug:                            │
│    → Look up its linked test cases in master_dataset        │
│  Aggregate all candidate test cases                         │
│  Score by weighted voting (similarity × link frequency)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               STAGE 3 — RERANKING LAYER                     │
│                                                             │
│  Rerank candidates using BAAI/bge-reranker-large            │
│  Cross-encode (new bug text, candidate testcase text)       │
│  Optional: hybrid scoring with LightGBM metadata features  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    TOP-K RECOMMENDATIONS                    │
│              Ordered list of testcase IDs                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Component Responsibilities

| Component | Role | Key Technology |
|---|---|---|
| `azure_client.py` | API authentication and data extraction | Azure DevOps REST API |
| `build_index.py` | Embed bugs, build FAISS index | SentenceTransformers, FAISS |
| `retrieve_similar_bugs.py` | ANN retrieval of similar bugs | FAISS |
| `candidate_generator.py` | Aggregate + weight candidate test cases | Pandas |
| `reranker.py` | Cross-encoder reranking | BAAI/bge-reranker-large |
| `hybrid_ranker.py` | Metadata-aware scoring | LightGBM |
| `evaluate.py` | Compute Recall@K, MRR | Scikit-learn, Pandas |

---

## 4. Development Phases

### Phase 0 — Data Enrichment (Azure DevOps Extraction)

**Goal:** Extract, merge, and validate all raw data from Azure DevOps.

**Tasks:**
- Authenticate via PAT (Personal Access Token) using `python-dotenv`
- Extract all bug work items with full metadata fields
- Extract all test case work items with full metadata fields
- Extract historical bug↔testcase link relationships
- Merge into `master_dataset.csv`
- Clean text fields (strip HTML, normalize whitespace)
- Validate schema completeness and referential integrity

**Deliverables:**

| File | Description |
|---|---|
| `data/bugs.csv` | All historical bugs with metadata |
| `data/testcases.csv` | All test cases with metadata |
| `data/master_dataset.csv` | Merged bug↔testcase mapping table |

---

### Phase 1 — Dataset Analysis (EDA)

**Goal:** Understand data quality, distribution, and potential failure modes before modeling.

**Tasks:**
- Missing value analysis per field
- Duplicate detection (duplicate `bug_id`, duplicate text)
- Long-tail analysis of testcase link frequency
- Text quality audit (empty titles, short descriptions, HTML artifacts)
- Metadata cardinality analysis (`area_path`, `priority`, `tags`)
- Overlap analysis between bug and testcase area paths

**Deliverables:**

| File | Description |
|---|---|
| `notebooks/eda.ipynb` | Full exploratory analysis notebook |
| `docs/eda_report.md` | Written summary of findings and data quality issues |

---

### Phase 2 — Baseline Retrieval System

**Goal:** Build a semantic bug retrieval system using dense embeddings and FAISS.

**Model:** `intfloat/e5-large-v2`

**Tasks:**
- Construct bug text representation: `[title] + [description]` (with e5 query/passage prefixes)
- Compute and persist bug embeddings
- Build FAISS `IndexFlatIP` (inner product / cosine) index over historical bugs
- Given a new bug query, retrieve Top-N similar historical bugs
- Evaluate retrieval quality: Recall@K on held-out split

**Deliverables:**

| File | Description |
|---|---|
| `src/build_index.py` | Embedding computation + FAISS index construction |
| `src/retrieve_similar_bugs.py` | Query interface for similar bug retrieval |
| `models/bug_faiss.index` | Serialized FAISS index |
| `models/bug_embeddings.npy` | Stored bug embedding matrix |

**Metrics:** Recall@1, Recall@5, Recall@10 on retrieval stage.

---

### Phase 3 — Candidate Testcase Generation

**Goal:** Translate retrieved similar bugs into a ranked pool of candidate test cases.

**Tasks:**
- For each retrieved bug (with similarity score `s_i`), fetch its linked test cases
- Aggregate all candidates across the Top-N retrieved bugs
- Apply **weighted voting**: each test case accumulates a score = Σ(s_i × link_weight)
- Deduplicate candidates and sort by aggregated score
- Output: ordered candidate list of (testcase_id, candidate_score) pairs

**Deliverables:**

| File | Description |
|---|---|
| `src/candidate_generator.py` | Full candidate aggregation and scoring logic |

---

### Phase 4 — Evaluation & Failure Analysis

**Goal:** Rigorously measure system performance and identify systematic failure modes.

**Tasks:**
- Split historical mappings into train/test sets (time-based split preferred)
- Run full pipeline on test bugs; compare predictions against ground-truth links
- Compute Recall@1, Recall@5, Recall@10, MRR
- Identify retrieval failures: bugs where no ground-truth test case was retrieved
- Analyze sparse test case coverage (test cases with very few historical links)
- Categorize failure modes (text quality, cold-start, domain mismatch)

**Deliverables:**

| File | Description |
|---|---|
| `src/evaluate.py` | Evaluation script with all metric computations |
| `docs/failure_analysis.md` | Structured failure report with examples |

---

### Phase 5 — Reranking Layer

**Goal:** Improve ranking precision by cross-encoding (bug, test case) pairs.

**Model:** `BAAI/bge-reranker-large`

**Tasks:**
- For each (new bug, candidate test case) pair: compute cross-encoder relevance score
- Re-sort candidate list by reranker score
- Evaluate improvement in Recall@1 and Recall@5 vs. Phase 3 baseline
- Tune reranking candidate pool size (trade-off: latency vs. recall)

**Deliverables:**

| File | Description |
|---|---|
| `src/reranker.py` | Cross-encoder reranking module |

---

### Phase 6 — Embedding Fine-Tuning

**Goal:** Adapt the retrieval embedding model to the domain-specific bug/testcase vocabulary.

**Framework:** `sentence-transformers` with `MultipleNegativesRankingLoss`

**Tasks:**
- Generate positive training pairs: (bug_text, testcase_text) from master_dataset
- Mine hard negatives: semantically similar but unlinked (bug, testcase) pairs via FAISS
- Fine-tune `intfloat/e5-large-v2` (or a smaller variant) on domain data
- Re-evaluate retrieval Recall@K with fine-tuned model vs. off-the-shelf baseline

**Deliverables:**

| File | Description |
|---|---|
| `src/finetune_embeddings.py` | Fine-tuning script with data prep and training loop |
| `models/finetuned_embedder/` | Saved fine-tuned model checkpoint |

---

### Phase 7 — Hybrid Metadata Ranking

**Goal:** Incorporate structured metadata signals into the ranking layer using a learned model.

**Tasks:**
- Engineer ranking features: area_path match, priority match, tag overlap, component similarity, recency signals
- Construct pointwise training dataset: (bug_id, testcase_id, feature_vector, label)
- Train `LightGBM` ranker (LambdaRank or pointwise regression)
- Combine LightGBM score with semantic similarity score (weighted fusion)
- Evaluate hybrid vs. semantic-only ranking

**Deliverables:**

| File | Description |
|---|---|
| `src/hybrid_ranker.py` | Feature extraction + LightGBM ranking module |
| `models/lgbm_ranker.pkl` | Trained LightGBM model artifact |

---

### Phase 8 — Advanced Research (Optional)

**Goal:** Explore graph-based learning to capture indirect bug↔testcase relationships.

**Research Directions:**
- Model the bug↔testcase historical graph as a heterogeneous bipartite graph
- Train a Graph Neural Network (GNN) — e.g., GraphSAGE or LightGCN — to learn node embeddings
- Use GNN embeddings as additional ranking features or standalone retrieval signals
- Explore co-occurrence patterns between test cases across shared bugs

**Note:** This phase is exploratory. Proceed only if Phases 2–7 leave meaningful headroom on evaluation metrics.

---

## 5. Coding Standards

All source code in `src/` must conform to the following standards.

### 5.1 Structure

- Each module has a **single, well-defined responsibility** (retrieval, generation, reranking, evaluation — never combined).
- Shared utilities (text cleaning, embedding helpers, logging setup) live in `src/utils.py`.
- No business logic in notebooks — notebooks call `src/` functions.

### 5.2 Configuration & Secrets

- All credentials (PAT tokens, organization URLs, project names) are loaded via `python-dotenv` from a `.env` file.
- `.env` is listed in `.gitignore` and **never committed**.
- All tuneable hyperparameters (Top-K, model names, batch sizes, thresholds) are defined in a `config.py` or a `config.yaml` file — never hardcoded inline.

```python
# Correct
from dotenv import load_dotenv
import os
load_dotenv()
PAT = os.getenv("AZURE_PAT")

# Wrong — never do this
PAT = "my_secret_token_abc123"
```

### 5.3 Type Hints

All function signatures must use Python type hints.

```python
def retrieve_similar_bugs(query_embedding: np.ndarray, top_k: int = 10) -> list[int]:
    ...
```

### 5.4 Logging

Use Python's `logging` module (not `print`) for all runtime output.

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Retrieved %d candidate bugs.", len(candidates))
```

### 5.5 Error Handling

- API calls must be wrapped in `try/except` with specific exception types.
- Failures must log the error with context before re-raising or gracefully continuing.
- Empty results (no candidates retrieved) must be handled explicitly — never silently return an incorrect fallback.

### 5.6 Reusability

- Functions must accept inputs and return outputs explicitly — no reliance on global state.
- File paths are always passed as arguments, never hardcoded inside functions.
- Embedding and index loading is cached where possible (load once, reuse across queries).

---

## 6. Evaluation Metrics

### 6.1 Metrics Used

#### Recall@K

For a given bug, Recall@K measures whether at least one ground-truth test case appears in the Top-K predictions.

```
Recall@K = (Number of bugs where ≥1 ground-truth TC appears in Top-K) / (Total bugs)
```

Reported at K = 1, 5, 10.

#### Mean Reciprocal Rank (MRR)

MRR rewards systems that place the correct test case **higher** in the ranked list.

```
MRR = (1/|Q|) × Σ (1 / rank_i)
```

where `rank_i` is the position of the first ground-truth test case for query `i`.

### 6.2 Why Accuracy Alone Is Insufficient

Standard classification accuracy is inappropriate here for several reasons:

- **Multi-label nature:** Each bug can have multiple correct test cases; accuracy cannot capture partial correctness.
- **Label imbalance:** With ~33k test cases, a classifier predicting the most common label achieves misleadingly high accuracy on frequent test cases while failing on sparse ones.
- **Ranking matters:** In practice, engineers act on the top few recommendations. A system that places the correct test case at rank 500 is operationally useless even if it technically "predicts" it.
- **Recall@K directly measures operational value:** Whether the correct test case appears in the shortlist an engineer reviews is the real-world success criterion.

### 6.3 Evaluation Split Strategy

- Prefer a **time-based split**: bugs created before date `T` form the training set; bugs after `T` form the test set.
- This simulates real deployment conditions where the system must generalize to future bugs.
- Avoid random splits that leak future test cases into training history.

---

## 7. Current Project Status

### Active Phase

**Phase 0 — Data Extraction**

### Completed

- [x] Azure DevOps Personal Access Token (PAT) generated and stored in `.env`
- [x] Project scaffold created (`data/`, `models/`, `notebooks/`, `src/`)
- [x] `src/azure_client.py` implemented with PAT authentication
- [x] Bug extraction script created
- [x] Test case extraction script created
- [x] Merge pipeline for `master_dataset.csv` created

### In Progress

- [ ] Run full extraction pipeline against Azure DevOps
- [ ] Validate schema completeness and row counts for all three output files
- [ ] Confirm referential integrity between `master_dataset.csv` and `bugs.csv` / `testcases.csv`

### Next Immediate Task

> **Run extraction pipeline and validate outputs.**
> Confirm `bugs.csv`, `testcases.csv`, and `master_dataset.csv` are populated with expected schemas and no critical missing fields before proceeding to Phase 1 EDA.

### Upcoming Milestones

| Phase | Status | Target |
|---|---|---|
| Phase 0 — Data Extraction | 🔄 In Progress | — |
| Phase 1 — EDA | ⏳ Pending | After Phase 0 validation |
| Phase 2 — Baseline Retrieval | ⏳ Pending | After Phase 1 |
| Phase 3 — Candidate Generation | ⏳ Pending | After Phase 2 |
| Phase 4 — Evaluation | ⏳ Pending | After Phase 3 |
| Phase 5 — Reranking | ⏳ Pending | After Phase 4 |
| Phase 6 — Fine-Tuning | ⏳ Pending | After Phase 5 |
| Phase 7 — Hybrid Ranking | ⏳ Pending | After Phase 6 |
| Phase 8 — Graph Research | ⏳ Optional | TBD |

---

*Last updated: Phase 0 — Data Extraction in progress.*
