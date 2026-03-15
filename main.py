from crewai import Crew, Process
from agents import job_analyst, candidate_retriever, candidate_evaluator, report_writer
from tasks import build_tasks
from dotenv import load_dotenv
import os

load_dotenv()

# ── INPUTS ───────────────────────────────────────────────────────────────────
# In your demo, these two variables are what the HR manager provides.
# Everything else is automated.

JOB_DESCRIPTION = """
We are looking for a Senior Data Engineer to join our fintech data platform team.

Requirements:
- 4+ years of experience in data engineering
- Strong proficiency in Python and SQL
- Experience with data pipelines (Apache Airflow, Spark, or similar)
- Familiarity with cloud platforms (AWS, GCP, or Azure)
- Experience with data warehousing (Snowflake, BigQuery, or Redshift)
- Understanding of data modeling and ETL/ELT processes

Nice to have:
- Experience in the financial services industry
- Knowledge of dbt (data build tool)
- Bachelor's degree in Computer Science, Engineering, or related field

The role involves designing and maintaining scalable data infrastructure,
collaborating with data scientists and analysts, and ensuring data quality
across the organization.
"""

PROFESSION = "ENGINEERING"  # must match one of your 24 directory names in Qdrant


# ── RUN ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AI HR Resume Screener")
    print("  Powered by CrewAI + Qdrant RAG")
    print("=" * 60)

    tasks = build_tasks(
        job_description=JOB_DESCRIPTION,
        profession=PROFESSION,
    )

    crew = Crew(
        agents=[job_analyst, candidate_retriever, candidate_evaluator, report_writer],
        tasks=tasks,
        process=Process.sequential,  # tasks run in order, each feeds the next
        verbose=True,
    )

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("  FINAL HIRING REPORT")
    print("=" * 60)
    print(result)

    # Save report to file
    with open("hiring_report.md", "w", encoding="utf-8") as f:
        f.write(str(result))
    print("\nReport saved to hiring_report.md")


if __name__ == "__main__":
    main()
