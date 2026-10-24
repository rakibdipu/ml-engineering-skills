"""
Incident Triage Pipeline with Structured LLM Output & Semantic Branching
Author: MD. RaKib Hassan Dipu (https://github.com/rakibdipu)
"""

from datetime import datetime
from pydantic import BaseModel, Field
from airflow.decorators import dag, task
from airflow.providers.common.ai.decorators import llm, llm_branch


class IncidentReport(BaseModel):
    severity: str = Field(description="'critical', 'warning', or 'info'")
    summary: str = Field(description="One sentence summary of the incident")
    recommended_action: str = Field(description="Steps the on-call engineer should take")


@dag(
    dag_id="incident_triage_pipeline",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ai", "ops", "llm"],
)
def incident_triage_pipeline():

    @task
    def receive_log_alert() -> str:
        return (
            "FATAL: Database pool exhausted in Auth Service. "
            "Connection timeout after 30000ms. 450 requests dropped."
        )

    # 1. Structured inference with Pydantic output validation
    @llm.task(
        model="gpt-4o-mini",
        conn_id="openai_default",
        system_prompt="You are an expert SRE on-call engineer analyzing server logs.",
        output_type=IncidentReport,
        temperature=0.0,
    )
    def analyze_incident(log_text: str) -> str:
        return f"Analyze the following production failure and categorize severity:\n\n{log_text}"

    # 2. Semantic DAG Branching based on LLM classification
    @llm_branch.task(
        model="gpt-4o-mini",
        conn_id="openai_default",
        system_prompt=(
            "Return EXACTLY one task ID string without formatting: "
            "'escalate_pagerduty' if severity is critical, otherwise 'post_slack_notification'."
        ),
        temperature=0.0,
    )
    def branch_by_severity(report: IncidentReport) -> str:
        return f"Determine routing for incident severity: {report.severity}"

    @task
    def escalate_pagerduty(report: IncidentReport):
        print(f"[PAGERDUTY ALARM] CRITICAL INCIDENT: {report.summary}")
        print(f"Action: {report.recommended_action}")

    @task
    def post_slack_notification(report: IncidentReport):
        print(f"[SLACK NOTICE] {report.severity.upper()}: {report.summary}")

    # DAG flow wiring
    log_data = receive_log_alert()
    report = analyze_incident(log_data)
    route = branch_by_severity(report)

    route >> [escalate_pagerduty(report), post_slack_notification(report)]


dag_instance = incident_triage_pipeline()
