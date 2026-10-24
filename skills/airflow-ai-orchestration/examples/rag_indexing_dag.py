"""
RAG Embedding Index Pipeline using TaskFlow Dynamic Mapping
Author: MD. RaKib Hassan Dipu (https://github.com/rakibdipu)
"""

from datetime import datetime
from airflow.decorators import dag, task
from airflow.providers.common.ai.decorators import embed


@dag(
    dag_id="rag_indexing_pipeline",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ai", "rag", "embeddings"],
)
def rag_indexing_pipeline():

    @task
    def fetch_unindexed_documents() -> list[str]:
        return [
            "TimesFM provides zero-shot forecasting for univariate and multivariate time series.",
            "SCARF uses random feature corruption for tabular contrastive representation learning.",
            "Airflow TaskFlow decorators simplify LLM agents, embeddings, and semantic branching.",
        ]

    # Dynamic parallel embedding generation
    @embed.task(
        model="text-embedding-3-small",
        conn_id="openai_default",
    )
    def generate_embedding(doc_chunk: str) -> list[float]:
        return doc_chunk

    @task
    def upsert_to_vector_database(embeddings: list[list[float]]):
        print(f"Successfully processed {len(embeddings)} document vectors.")
        for i, vector in enumerate(embeddings):
            print(f"Vector {i+1} dimensions: {len(vector)}")

    docs = fetch_unindexed_documents()
    # Dynamic task mapping (.expand) calls embedding generation in parallel across worker nodes
    vectors = generate_embedding.expand(doc_chunk=docs)
    upsert_to_vector_database(vectors)


dag_instance = rag_indexing_pipeline()
