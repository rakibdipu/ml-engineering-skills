# 🚀 ML Engineering Skills & Agent Extensions

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://python.org)
[![Platform: Antigravity / Claude Code / AI Agents](https://img.shields.io/badge/AI%20Agents-Antigravity%20%7C%20Claude%20Code-orange.svg)](#)
[![Maintained by: rakibdipu](https://img.shields.io/badge/Maintained%20by-rakibdipu-purple.svg)](https://github.com/rakibdipu)

A production-hardened suite of **AI Agent Skills, Runbooks, and Reference Pipelines** bridging state-of-the-art machine learning research into practical engineering workflows.

This repository curates, resolves architectural gaps in, and standardizes three powerful ML technologies:
1. **Google Research's TimesFM** (Time Series Foundation Model)
2. **SCARF** (Self-Supervised Contrastive Learning for Tabular Data)
3. **Apache Airflow AI Orchestration** (Native TaskFlow GenAI & Agent Pipelines)

---

## 👨‍💻 Project Origin, Credits & Improvements

### 🌟 Original Research & Creators Credit
We gratefully acknowledge the original authors and maintainers whose breakthrough work laid the foundation:
* **TimesFM:** Developed by the **Google Research** team (*Das et al.*). Original repository: [`google-research/timesfm`](https://github.com/google-research/timesfm).
* **SCARF:** Based on the paper *"SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption"* by **Dara Bahri, Heinrich Jiang, Gitit Levin, Sven Gowal** (Google Research, ICLR 2022). PyTorch implementation by **Corentin Labrugere** ([`clabrugere/pytorch-scarf`](https://github.com/clabrugere/pytorch-scarf)).
* **Airflow AI Orchestration:** Conceptualized by **Astronomer** ([`astronomer/airflow-ai-sdk`](https://github.com/astronomer/airflow-ai-sdk)) and officially upstreamed by the **Apache Airflow Community** ([`apache-airflow-providers-common-ai`](https://github.com/apache/airflow)).

---

### 🛠️ What I Improved & Enhanced (by MD. RaKib Hassan Dipu)
While the original repositories offer great core algorithms, they were siloed, lacked production-ready agent skills, had breaking namespace/dependency shifts, or missed end-to-end downstream pipelines. 

**Key improvements introduced in this repository:**

### 1. Google TimesFM (`timesfm-forecasting`)
* **Dynamic Checkpoint Loading:** Eliminated fragile hardcoded model dimension/layer configurations; the loader now automatically resolves model hyperparameters directly from Hugging Face checkpoint weights.
* **Multi-Series DataFrame Forecasting:** Added complete multi-store / multi-item Pandas DataFrame pipelines (`forecast_on_df`) with quantile prediction intervals ($10^{th}$ to $90^{th}$ percentile bands) for risk management.
* **Covariate Integration:** Documented and added examples for future covariates (holidays, promotion discounts) and past covariates.
* **Enterprise Scaling:** Added reference implementations for domain-specific fine-tuning and Google Cloud BigQuery ML (`AI.FORECAST`) zero-infrastructure deployment.

### 2. Tabular SCARF (`tabular-scarf`)
* **Namespace & Import Fixes:** Fixed broken module imports to match the official PyPI package (`pytorch_scarf`).
* **Real-World Mixed-Type Preprocessing:** Raw SCARF only handles purely numeric matrices. Added end-to-end preprocessing pipelines (`ColumnTransformer` with `StandardScaler` for continuous features and `OrdinalEncoder` for categorical columns).
* **Downstream Gradient Boosting Pipeline:** Built an automated representation extraction pipeline that feeds frozen SCARF encoder embeddings directly into gradient-boosted trees (**XGBoost / LightGBM**), yielding significant accuracy jumps on small labeled datasets.
* **Tuning Playbook:** Added clear hyperparameter guidelines based on empirical testing (corruption rate optimal at `0.6`, temperature at `0.07`, batch size $\ge 256$).

### 3. Airflow AI Orchestration (`airflow-ai-orchestration`)
* **Migration to Official Apache Standard:** Astronomer's `airflow-ai-sdk` was deprecated. Fully migrated and refactored all patterns to the canonical `apache-airflow-providers-common-ai` standard.
* **Deterministic Semantic Branching:** Fixed non-deterministic DAG routing errors in `@task.llm_branch` by enforcing strict temperature bounds (`temperature=0.0`) and exact task ID schemas.
* **RAG Pipeline Architecture:** Added complete vector embedding generation and parallel ingestion DAGs using TaskFlow dynamic mapping (`@task.embed` + `.expand()`).
* **Production Guardrails:** Added secret management through Airflow Connections, API rate-limit retry logic (`retries=3`), token budgeting, and agent step bounds (`max_steps=10`) to eliminate runaway LLM bills.

### 4. Agentic AI Ecosystem Integration
* Packaged everything as **drop-in Skills** compatible with **Antigravity**, **Claude Code**, and **Gemini CLI**.
* Provided a **1-command installer script** (`scripts/install_plugin.sh`) and standalone runnable Python examples for every skill.

---

## 📂 Repository Structure

```text
ml-engineering-skills/
├── plugin.json                              # Antigravity/Gemini agent plugin manifest
├── requirements.txt                         # Unified Python dependencies
├── LICENSE                                  # MIT License (MD. RaKib Hassan Dipu)
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

### Option 1: Install into Antigravity / Gemini CLI / Claude Code

Register this plugin and all skills globally into your AI coding assistant:

```bash
git clone https://github.com/rakibdipu/ml-engineering-skills.git
cd ml-engineering-skills
./scripts/install_plugin.sh
```

This installs the bundle into `~/.gemini/config/plugins/ml-engineering/`. The agent will automatically detect and load these skills whenever relevant tasks are prompted.

### Option 2: Standalone Python Usage

Clone the repository and install the dependencies:

```bash
git clone https://github.com/rakibdipu/ml-engineering-skills.git
cd ml-engineering-skills
pip install -r requirements.txt
```

Run any of the included examples:

```bash
# 1. TimesFM zero-shot prediction
python skills/timesfm-forecasting/examples/quickstart_zero_shot.py

# 2. Multi-series sales forecasting on Pandas DataFrames
python skills/timesfm-forecasting/examples/dataframe_forecasting.py

# 3. Pretrain SCARF contrastive encoder on tabular data
python skills/tabular-scarf/examples/pretrain_scarf.py

# 4. Evaluate extracted embeddings with XGBoost
python skills/tabular-scarf/examples/downstream_xgboost.py
```

---

## 📖 Skills Reference Summary

### 1. [`timesfm-forecasting`](skills/timesfm-forecasting/SKILL.md)
* **When to use:** Zero-shot forecasting for retail demand, energy loads, server traffic, financial indicators, or weather without manual model training.
* **Key Checkpoint:** `google/timesfm-1.0-200m-pytorch` or `google/timesfm-2.0-500m-pytorch`.

### 2. [`tabular-scarf`](skills/tabular-scarf/SKILL.md)
* **When to use:** Datasets where you have large volumes of unlabeled tabular/CSV rows and only a few labeled targets. 
* **Key Mechanism:** SimCLR for tabular data via random feature corruption from empirical marginal distributions + NT-Xent contrastive loss.

### 3. [`airflow-ai-orchestration`](skills/airflow-ai-orchestration/SKILL.md)
* **When to use:** Automating multi-step AI agents, LLM evaluations, RAG vector indexing, and conditional branching inside Apache Airflow data pipelines.
* **Key Package:** `apache-airflow-providers-common-ai` (decorators: `@task.llm`, `@task.agent`, `@task.embed`, `@task.llm_branch`).

---

## 📄 License & Attribution

* This project is licensed under the [MIT License](LICENSE) - maintained by **[MD. RaKib Hassan Dipu](https://github.com/rakibdipu)**.
* Individual algorithmic designs and underlying architectures belong to their respective original authors and organizations as credited above.
