# TariffMind — Zero-Hallucination HS 2022 Customs Classifier

**TariffMind** is a domain-closed customs classification engine that maps messy commercial invoice descriptions to legally structured 6-digit **Harmonized System (HS 2022)** tariff codes.

It combines **ChromaDB RAG** over the official WCO HS 2022 nomenclature, a **QLoRA fine-tuned IBM Granite 3.3** model for General Rules of Interpretation (GRI) reasoning, and a **deterministic HS tree validator** that guarantees **0.0% hallucinated codes**.

---

## Benchmark Results (Chapters 84–85: Electronics & Machinery)

Evaluated on **138 held-out test samples** (`data/eval_gri.jsonl`) including underspecified invoice items:

| Metric | Score |
| :--- | :---: |
| **JSON Schema Validity** | **100.0%** |
| **Hallucination Rate** | **0.0%** |
| **2-Digit Accuracy (Chapter)** | **100.0%** |
| **4-Digit Accuracy (Heading)** | **100.0%** |
| **6-Digit Exact Match (Subheading)** | **97.44%** |
| **Abstention Accuracy (Human Review)** | **100.0%** |

---

## Architecture

1. **Candidate Retrieval (`src/build_rag.py`):** Embeds all 5,613 WCO HS 2022 subheadings (chained with Chapter and Heading context) into ChromaDB using `BAAI/bge-small-en-v1.5` and retrieves the top-10 candidates per query.
2. **Fine-Tuned Reasoning (`src/train_granite.py`):** 4-bit QLoRA fine-tune of `unsloth/granite-3.3-2b-instruct` (supports `8b` via `--model`) trained to apply GRI rules and output strict JSON—or abstain (`needs_human_review: true`) when critical technical specs are missing.
3. **Deterministic Guardrail (`src/validator.py`):** Validates every predicted 6-digit code against `data/hs_2022.csv` (blocking catch-all Chapter 99) and reconstructs the official Section → Chapter → Heading → Subheading hierarchy.
4. **Interactive UI (`app.py`):** Streamlit app supporting single-item classification, batch commercial invoice CSV processing, and a live evaluation dashboard.

---

## Project Structure

```text
tariffmind/
├── data/
│   ├── hs_2022.csv            # Official WCO HS 2022 hierarchy
│   ├── train_gri.jsonl        # Fine-tuning dataset (782 samples)
│   ├── eval_gri.jsonl         # Held-out evaluation set (138 samples)
│   └── eval_results.json      # Benchmark output
├── src/
│   ├── build_rag.py           # ChromaDB HS 2022 indexer & retriever
│   ├── validator.py           # Deterministic zero-hallucination validator
│   ├── generate_dataset.py    # Messy invoice + abstention dataset generator
│   ├── train_granite.py       # Unsloth 4-bit QLoRA fine-tuning + GGUF export
│   └── eval.py                # Hierarchical evaluation harness
└── app.py                     # Streamlit web interface
