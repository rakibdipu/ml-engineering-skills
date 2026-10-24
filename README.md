# 🚀 ML Engineering Skills & Agent Extensions

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://python.org)
[![Platform: Antigravity / Claude Code / AI Agents](https://img.shields.io/badge/AI%20Agents-Antigravity%20%7C%20Claude%20Code-orange.svg)](#)

Curated, production-hardened AI Agent skills and reference architectures for modern Machine Learning Engineering workflows:
1. **Google Research's TimesFM** (Time Series Foundation Model)
2. **SCARF** (Self-Supervised Contrastive Learning for Tabular Data)
3. **Apache Airflow AI Orchestration** (Successor to Astronomer's `airflow-ai-sdk` & `apache-airflow-providers-common-ai`)

Maintained and authored by **[MD. RaKib Hassan Dipu](https://github.com/rakibdipu)**.

---

## 📌 Included Skills Overview

| Skill | Based On | Domain | Primary Capability |
| :--- | :--- | :--- | :--- |
| **[`timesfm-forecasting`](skills/timesfm-forecasting/SKILL.md)** | [google-research/timesfm](https://github.com/google-research/timesfm) | Time Series Forecasting | Zero-shot & fine-tuned multi-horizon forecasting, covariate integration, BigQuery ML |
| **[`tabular-scarf`](skills/tabular-scarf/SKILL.md)** | [clabrugere/pytorch-scarf](https://github.com/clabrugere/pytorch-scarf) | Tabular Data / SSL | Random feature corruption contrastive learning, dense embedding extraction for XGBoost |
| **[`airflow-ai-orchestration`](skills/airflow-ai-orchestration/SKILL.md)** | [astronomer/airflow-ai-sdk](https://github.com/astronomer/airflow-ai-sdk) & `common-ai` | MLOps & GenAI Orchestration | Native Airflow TaskFlow decorators (`@task.llm`, `@task.agent`, `@task.embed`, `@task.llm_branch`) |

---

## 🔍 Key Gaps Resolved & Enhancements Over Raw Repos

* **TimesFM:**
  * Resolved hyperparameter misalignment by delegating hidden dimension and layer resolution to pretrained Hugging Face checkpoint weights rather than static hardcoding.
  * Added complete multi-series Pandas DataFrame forecasting (`forecast_on_df`) with quantile intervals.
  * Added domain fine-tuning and Google Cloud BigQuery ML (`AI.FORECAST`) enterprise patterns.

* **SCARF (Tabular Contrastive Learning):**
  * Corrected Python package import paths (`from pytorch_scarf.model import SCARF`).
  * Added preprocessing pipeline for mixed numerical-categorical tabular datasets (`StandardScaler` + `OrdinalEncoder`).
  * Provided complete downstream classification pipeline extracting representations from `model.encoder` into XGBoost.
  * Documented temperature scaling (`0.07`), corruption rates (`0.6`), and batch sizing pitfalls.

* **Airflow AI Orchestration:**
  * Addressed Astronomer's deprecation of `airflow-ai-sdk` by structuring code around the official upstream standard: `apache-airflow-providers-common-ai`.
  * Included complete working TaskFlow DAGs for **Structured LLM outputs (Pydantic)**, **Deterministic Semantic Branching (`@task.llm_branch`)**, and **RAG Embedding Ingestion (`@task.embed`)**.
  * Added production safety guards: rate limit retries, token budgeting, secret connection management, and agent step bounds (`max_steps`).

---

## 📦 Directory Structure

```text
ml-engineering-skills/
├── plugin.json                              # Antigravity/Gemini agent plugin manifest
├── requirements.txt                         # Unified Python dependencies
├── LICENSE                                  # MIT License
├── README.md                                # Comprehensive documentation
├── scripts/
│   └── install_plugin.sh                    # 1-command installer for local/global agent config
└── skills/
    ├── timesfm-forecasting/
    │   ├── SKILL.md                         # Detailed agent runbook & instructions
    │   └── examples/
    │       ├── quickstart_zero_shot.py      # Zero-shot forecasting script
    │       └── dataframe_forecasting.py     # Multi-store Pandas DataFrame forecasting
    ├── tabular-scarf/
    │   ├── SKILL.md                         # Tabular SSL contrastive learning runbook
    │   └── examples/
    │       ├── pretrain_scarf.py            # Pretraining script with NT-Xent loss
    │       └── downstream_xgboost.py        # Feature extraction + XGBoost classification
    └── airflow-ai-orchestration/
        ├── SKILL.md                         # Airflow AI TaskFlow runbook
        └── examples/
            ├── incident_triage_dag.py       # SRE incident triage DAG with semantic branching
            └── rag_indexing_dag.py          # Parallel document embedding ingestion DAG
```

---

## ⚡ Quickstart & Installation

### Option 1: Install into Antigravity / Gemini CLI (Recommended)

Run the included install script to register the plugin and all skills globally:

```bash
git clone https://github.com/rakibdipu/ml-engineering-skills.git
cd ml-engineering-skills
./scripts/install_plugin.sh
```

This copies the plugin to `~/.gemini/config/plugins/ml-engineering/`, automatically enabling all skills across your agent conversations.

### Option 2: Standalone Python Usage

Clone the repository and install the dependencies:

```bash
git clone https://github.com/rakibdipu/ml-engineering-skills.git
cd ml-engineering-skills
pip install -r requirements.txt
```

Run any of the included examples:

```bash
# TimesFM zero-shot prediction
python skills/timesfm-forecasting/examples/quickstart_zero_shot.py

# Pretrain SCARF on tabular data
python skills/tabular-scarf/examples/pretrain_scarf.py

# Evaluate embeddings with XGBoost
python skills/tabular-scarf/examples/downstream_xgboost.py
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

Developed with ❤️ by **[MD. RaKib Hassan Dipu](https://github.com/rakibdipu)**.
