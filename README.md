# HyperRAG

**HyperRAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented Generation**

> Accepted at **The Web Conference (WWW) 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2602.14470v1-b31b1b.svg)](https://arxiv.org/abs/2602.14470v1)
[![Python 3.11.13](https://img.shields.io/badge/python-3.11.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

HyperRAG addresses a fundamental limitation of conventional RAG systems: the inability to capture **N-ary facts** — relationships that involve more than two entities simultaneously (e.g., *"Person A received Award B from Organization C in Year D"*).

Instead of decomposing such facts into binary edges (losing relational context), HyperRAG encodes them as **hyperedges** in a hypergraph, where a single edge can connect any number of nodes. This structure enables:

- **Faithful representation** of complex, multi-entity facts without information loss
- **Structured multi-hop reasoning** by traversing hyperedges across the graph
- **Precise retrieval** via a trained MLP-based retriever (HyperRetriever) that scores candidate hyperedges given a query

The system consists of two core modules:

| Module | Role | Key Mechanism |
|---|---|---|
| **HyperMemory** | Memory-Guided Beam Retriever | Leverages the LLM’s parametric memory to guide beam search over n-ary facts without extra training. |
| **HyperRetriever** | Learnable Relational Retriever | Uses a trained MLP to fuse structural and semantic signals for adaptive, query-aware chain extraction. |

The codebase builds on [HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG.git).

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Datasets](#datasets)
4. [Project Structure](#project-structure)
5. [Pipeline: WikiTopics (Closed Domain)](#pipeline-wikitopics-closed-domain)
   - [Step 1 — Build the Hypergraph](#step-1--build-the-hypergraph)
   - [Step 2 — HyperMemory Inference](#step-2--hypermemory-inference)
   - [Step 3 — HyperRetriever Training](#step-3--hyperretriever-training)
   - [Step 4 — HyperRetriever Inference](#step-4--hyperretriever-inference)
6. [Pipeline: Open Domain](#pipeline-open-domain)
7. [Evaluation](#evaluation)
8. [License](#license)
9. [Citation](#citation)

---

## Installation

```bash
conda create -n hyperrag python=3.11.13
conda activate hyperrag
pip install -r requirements.txt
```

---

## Configuration

Create a `config.json` file in the project root with your API credentials:

```json
{
    "openai_api_key": "YOUR_OPENAI_API_KEY"
}
```

---

## Datasets

### WikiTopics (Closed Domain)

WikiTopics is a closed-domain multi-hop QA dataset organized into **11 topic domains**:

| Domain | Key | Domain | Key |
|---|---|---|---|
| Art | `art` | Infrastructure | `infra` |
| Award | `award` | Location | `loc` |
| Education | `edu` | Organization | `org` |
| Health | `health` | People | `people` |
| Science | `sci` | Sport | `sport` |
| Taxonomy | `tax` | | |

Each domain provides both a **Knowledge Graph (KG)** version and a **Natural Language (NLG)** version. The main method uses the NLG version; the KG version is used for ablation studies.

| Version | Download |
|---|---|
| Full Dataset — KG | [🔗 WikiTopics KG](https://reltrans.s3.us-east-2.amazonaws.com/WikiTopics_QE.zip) |
| Full Dataset — NLG | [🔗 WikiTopics NLG](https://drive.google.com/file/d/13xpzP1MCld5Cpr4W7YNyVkscD8mcKysC/view?usp=sharing) |
| Sampled Dataset (1%) | [`dataset/wikitopics_test_sampled`](dataset/wikitopics_test_sampled) |

After downloading the full WikiTopics dataset, place it in the `dataset/` folder:

```
dataset/
├── open_domain_dataset/
├── open_domain_splitted/
├── wikitopics_test_sampled/
└── WikiTopicsQE_NLG/          <-- place the full WikiTopics dataset here
```

### Open Domain

| Split | Path |
|---|---|
| Full dataset | [`dataset/open_domain_dataset`](dataset/open_domain_dataset) |
| Pre-split for training/testing | [`dataset/open_domain_splitted`](dataset/open_domain_splitted) |

Open domain includes: **2WikiMultiHopQA**, **HotpotQA**, and **MuSiQue**.

---

## Project Structure

```
.
├── dataset/
│   ├── open_domain_dataset/
│   ├── open_domain_splitted/
│   ├── wikitopics_test_sampled/
│   └── WikiTopicsQE_NLG/
├── evaluate/
│   ├── qa_eval_EM_F1.py              # Open domain evaluation
│   └── qa_eval_MRR_HIT.py            # Closed domain evaluation
├── HyperMemory/                      # Graph construction + memory-based QA (WikiTopics)
├── HyperMemory_open/                 # Memory-based QA (open domain)
├── HyperMemory_token/                # Token efficiency variant
├── HyperRetriever/                   # MLP retriever QA (WikiTopics)
├── HyperRetriever_open/              # MLP retriever QA (open domain)
├── HyperRetriever_token/             # Token efficiency variant
├── HyperRetriever_token_kg/          # Token efficiency variant (KG input)
└── results/                          # Auto-generated inference outputs
```

---

## Pipeline: WikiTopics (Closed Domain)

Replace `{DOMAIN}` with one of the 11 domain keys (e.g., `art`, `award`, `edu`, ...).

### Step 1 — Build the Hypergraph

This step constructs the hypergraph from the WikiTopics NLG corpus. The hypergraph encodes N-ary facts as hyperedges and is shared by both HyperMemory and HyperRetriever. Outputs are written to an `expr/` directory in the project root.

```bash
cd HyperMemory
python wikitopics_construct.py {DOMAIN}
```

> **Output:** `expr/{DOMAIN}/` — contains the hypergraph structure, node/edge embeddings, and associated index files used in all downstream steps.

---

### Step 2 — HyperMemory Inference

Run question answering directly over the constructed hypergraph using the memory-based approach. No additional training is required.

```bash
# From HyperMemory/
python wikitopics_query.py {DOMAIN}
```

> **Output:** `results/HyperMemory/{DOMAIN}_output.jsonl`

---

### Step 3 — HyperRetriever Training

HyperRetriever improves retrieval precision by training a lightweight MLP on top of hypergraph embeddings. Complete both sub-steps before running inference.

#### 3a. Prepare Training Data

```bash
cd HyperRetriever
python retrieve/prepare.py {DOMAIN}
```

#### 3b. Train the MLP

```bash
python retrieve/train.py {DOMAIN}
```

> **Output:** A trained MLP checkpoint saved under `expr/` for the specified domain.

---

### Step 4 — HyperRetriever Inference

Run question answering using the trained retriever.

```bash
# From HyperRetriever/
python wikitopics_query.py {DOMAIN}
```

> **Output:** `results/HyperRetriever/{DOMAIN}_output.jsonl`

---

## Pipeline: Open Domain

The open domain pipeline follows the same logic as WikiTopics. Use the modules ending with `_open`.

---

## Evaluation

All inference scripts automatically write results to `results/{MODULE}/`, with filenames ending in `_output.jsonl`.

### Closed Domain — WikiTopics

Use **MRR (Mean Reciprocal Rank)** and **Hit Rate** to evaluate answer ranking quality:

```bash
python evaluate/qa_eval_MRR_HIT.py --model {OUTPUT_FOLDER} {DATASET}
```

**Example:**
```bash
python evaluate/qa_eval_MRR_HIT.py --model HyperMemory art
```

### Open Domain — 2Wiki / HotpotQA / MuSiQue

Use **Exact Match (EM)** and **F1 Score** to evaluate answer extraction quality:

```bash
python evaluate/qa_eval_EM_F1.py --model_name {OUTPUT_FOLDER} {DATASET}
```

### Metric Summary

| Dataset Type | Metric | Script |
|---|---|---|
| Closed domain (WikiTopics) | MRR, Hit Rate | `qa_eval_MRR_HIT.py` |
| Open domain (2Wiki, HotpotQA, MuSiQue) | Exact Match, F1 | `qa_eval_EM_F1.py` |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use HyperRAG in your research, please cite:

```bibtex
@inproceedings{lien2026hyperrag,
    title={HyperRAG: Reasoning N-ary Facts over Hypergraphs for Retrieval Augmented Generation},
    author={Wen-Sheng Lien, Yu-Kai Chan, Hao-Lung Hsiao, Bo-Kai Ruan, Meng-Fen Chiang, Chien-An Chen, Yi-Ren Yeh and Hong-Han Shuai},
    booktitle={The Web Conference (WWW)},
    year={2026}
}
```

---

## TODO

- [v] arXiv Paper link
- [v] Installation Instructions
- [ ] Integrate token counter function into modules