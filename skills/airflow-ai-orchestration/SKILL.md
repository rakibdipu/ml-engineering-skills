---
name: airflow-ai-orchestration
description: >-
  Orchestrate LLM workflows, AI agents, RAG pipelines, text embeddings, and
  semantic DAG branching in Apache Airflow using TaskFlow decorators. Covers the
  official `apache-airflow-providers-common-ai` package (successor to the
  deprecated `astronomer/airflow-ai-sdk`) with `@task.llm`, `@task.agent`,
  `@task.embed`, and `@task.llm_branch` patterns. Use when adding generative AI
  or autonomous agent steps into production Airflow data pipelines.
---

# Airflow AI Orchestration

> **Source:** [astronomer/airflow-ai-sdk](https://github.com/astronomer/airflow-ai-sdk) → now upstreamed to [apache/airflow-providers-common-ai](https://github.com/apache/airflow)  
> **Built on:** [Pydantic AI](https://docs.pydantic.dev/latest/concepts/pydantic_ai/) + Apache Airflow TaskFlow API

> [!IMPORTANT]
> `astronomer/airflow-ai-sdk` is **officially deprecated**. The canonical package is now:
> ```bash
> pip install apache-airflow-providers-common-ai
> ```
> The decorator API (`@task.llm`, `@task.agent`, etc.) is identical in both.

---

## 1. Installation

```bash
# Airflow 2.8+ required
pip install "apache-airflow>=2.8"

# Official AI provider
pip install apache-airflow-providers-common-ai

# LLM provider SDKs (install what you need)
pip install openai           # OpenAI GPT-4o, gpt-4o-mini
pip install anthropic        # Claude 3.5 Sonnet, Haiku
pip install google-genai     # Gemini 2.0

# Legacy (only for maintaining old Astronomer repos)
pip install airflow-ai-sdk
```

### Airflow Connection Setup (Secrets Management)

Never hardcode API keys. Set up Airflow Connections:

```bash
# Via Airflow CLI
airflow connections add openai_default \
  --conn-type generic \
  --conn-password "sk-..."

airflow connections add anthropic_default \
  --conn-type generic \
  --conn-password "sk-ant-..."

# Or via environment variables (picked up automatically)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## 2. Core Decorator Reference

| Decorator | Import | Purpose |
| :--- | :--- | :--- |
| `@task.llm` | `airflow.providers.common.ai.decorators.llm` | Single structured LLM call with Pydantic output schema |
| `@task.agent` | `airflow.providers.common.ai.decorators.agent` | Multi-step autonomous agent loop with tool calling |
| `@task.embed` | `airflow.providers.common.ai.decorators.embed` | Batch text → embedding vectors (for RAG / vector DBs) |
| `@task.llm_branch` | `airflow.providers.common.ai.decorators.llm_branch` | Semantic DAG branching — LLM returns task ID string |

---

## 3. Structured LLM Task (`@task.llm`)

The function body returns the **prompt string**. The decorator executes the call and parses the output into a Pydantic model.

```python
from datetime import datetime
from pydantic import BaseModel, Field
from airflow.decorators import dag, task
from airflow.providers.common.ai.decorators import llm

class SentimentResult(BaseModel):
    label: str        = Field(description="'positive', 'negative', or 'neutral'")
    score: float      = Field(description="Confidence score 0.0 to 1.0")
    explanation: str  = Field(description="One-sentence reasoning")

@dag(schedule="@hourly", start_date=datetime(2025, 1, 1), catchup=False)
def review_sentiment_pipeline():

    @task
    def fetch_reviews() -> list[str]:
        # Simulate fetching reviews from DB / API
        return [
            "The product is absolutely fantastic!",
            "Terrible experience, never buying again.",
        ]

    @llm.task(
        model="gpt-4o-mini",
        conn_id="openai_default",           # Airflow Connection name for API key
        system_prompt="You are a precise sentiment analysis engine. Respond in JSON.",
        output_type=SentimentResult,        # Pydantic model — guarantees structured output
        temperature=0.0,                    # Set 0 for deterministic, reproducible DAG runs
        retries=3,
        retry_delay=30,
    )
    def analyze_sentiment(review: str) -> str:
        return f"Classify the sentiment of this customer review:\n\n{review}"

    @task
    def store_results(results: list[SentimentResult]):
        for r in results:
            print(f"{r.label} ({r.score:.2f}): {r.explanation}")

    reviews = fetch_reviews()
    # Dynamic task mapping — runs one LLM task per review in parallel
    sentiments = analyze_sentiment.expand(review=reviews)
    store_results(sentiments)

review_sentiment_pipeline()
```

---

## 4. Autonomous Agent with Tool Calling (`@task.agent`)

The agent loop runs until the model decides it has completed the task (or hits `max_steps`).

```python
from airflow.providers.common.ai.decorators import agent

# Tools are plain Python functions with docstrings (used as tool descriptions)
def get_stock_price(ticker: str) -> dict:
    """Fetch the latest stock price for a given ticker symbol."""
    # Replace with real API call (yfinance, Alpha Vantage, etc.)
    return {"ticker": ticker, "price": 182.47, "currency": "USD"}

def get_company_news(ticker: str, days: int = 7) -> list[str]:
    """Get the last N days of news headlines for a stock ticker."""
    return [f"{ticker}: Q3 earnings beat expectations by 8%"]

@agent.task(
    model="claude-3-5-sonnet-20241022",
    conn_id="anthropic_default",
    tools=[get_stock_price, get_company_news],
    system_prompt=(
        "You are a financial research assistant. "
        "Use the provided tools to research stocks and produce a brief summary report."
    ),
    max_steps=10,     # Hard limit on agent iterations to prevent runaway costs
    retries=2,
)
def research_stock(ticker: str) -> str:
    return f"Research {ticker} stock: get current price and recent news, then summarize."
```

---

## 5. Text Embeddings for RAG Pipelines (`@task.embed`)

Embed text chunks and upsert to a vector database for downstream retrieval:

```python
from airflow.providers.common.ai.decorators import embed

@dag(schedule="@daily", start_date=datetime(2025, 1, 1), catchup=False)
def rag_index_pipeline():

    @task
    def extract_documents() -> list[str]:
        # Simulate pulling docs from S3, database, or Confluence
        return [
            "Our refund policy allows returns within 30 days.",
            "Premium plans include 24/7 dedicated support.",
        ]

    @embed.task(
        model="text-embedding-3-small",     # OpenAI embedding model
        conn_id="openai_default",
    )
    def embed_documents(text: str) -> list[float]:
        return text   # embed.task wraps this and returns the embedding vector

    @task
    def upsert_to_vectordb(doc_embeddings: list[list[float]]):
        # Push to Pinecone / Qdrant / Chroma / pgvector
        for i, vector in enumerate(doc_embeddings):
            print(f"Upserting doc {i}: vector dim={len(vector)}")

    docs = extract_documents()
    vectors = embed_documents.expand(text=docs)   # Parallel embedding via dynamic task map
    upsert_to_vectordb(vectors)

rag_index_pipeline()
```

---

## 6. Semantic DAG Branching (`@task.llm_branch`)

Route DAG execution paths based on natural-language classification:

```python
from airflow.providers.common.ai.decorators import llm_branch

@dag(schedule=None, start_date=datetime(2025, 1, 1), catchup=False)
def support_ticket_router():

    @task
    def receive_ticket() -> str:
        return "My payment failed at checkout and I was charged twice."

    # llm_branch.task must return EXACTLY one downstream task_id string
    # The function body is the classification prompt
    @llm_branch.task(
        model="gpt-4o-mini",
        conn_id="openai_default",
        system_prompt=(
            "You are a ticket routing system. Return ONLY one of these exact values: "
            "'billing_team', 'technical_team', or 'general_support'. No other output."
        ),
        temperature=0.0,  # Critical: must be deterministic for branching
    )
    def classify_ticket(ticket: str) -> str:
        return f"Route this support ticket to the correct team:\n\n{ticket}"

    @task
    def billing_team(ticket: str):
        print(f"[BILLING] Processing: {ticket}")

    @task
    def technical_team(ticket: str):
        print(f"[TECH] Processing: {ticket}")

    @task
    def general_support(ticket: str):
        print(f"[GENERAL] Processing: {ticket}")

    ticket = receive_ticket()
    branch = classify_ticket(ticket)
    # All branch tasks are defined; only the one returned by the LLM will execute
    branch >> [billing_team(ticket), technical_team(ticket), general_support(ticket)]

support_ticket_router()
```

> [!WARNING]
> `@task.llm_branch` must return an **exact task ID string** (no markdown, no explanation).
> Always set `temperature=0.0` and enforce the constraint in the system prompt to prevent routing failures.

---

## 7. Production Best Practices

### Security
- **Never hardcode API keys.** Use Airflow Connections (`conn_id`) or environment variables.
- Rotate API keys via secret backends (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault).

### Reliability & Cost Control
- **Always set `max_steps`** on `@task.agent` to prevent runaway agent loops and unbounded costs.
- Use **cheaper models** (e.g., `gpt-4o-mini`, `claude-haiku`) for classification/extraction tasks; reserve frontier models for complex reasoning.
- Set **`retries=3, retry_delay=30`** on all LLM tasks to handle transient rate-limit errors (HTTP 429).

### Idempotency
- LLM outputs are non-deterministic. Set `temperature=0.0` for tasks used in branching or downstream computation.
- Use `output_type` Pydantic models to validate and guarantee the schema of LLM outputs before they flow to downstream tasks.

### Monitoring
```python
# Add SLA and on_failure_callback to catch LLM task hangs
@llm.task(
    model="gpt-4o-mini",
    sla=timedelta(minutes=5),
    on_failure_callback=lambda ctx: alert_slack(ctx),
)
def my_llm_task(data: str) -> str:
    return f"Process: {data}"
```

### Astro Runtime / Astronomer Cloud
If deploying on Astronomer, the `apache-airflow-providers-common-ai` provider is available in Astro Runtime `12.0+`. Add to your `requirements.txt`:

```
apache-airflow-providers-common-ai>=1.0.0
openai>=1.0.0
anthropic>=0.30.0
```
