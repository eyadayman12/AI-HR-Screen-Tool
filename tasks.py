from crewai import Task
from agents import job_analyst, candidate_retriever, candidate_evaluator, report_writer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

qdrant = QdrantClient("http://localhost:6333", timeout=60)
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
COLLECTION = "hr_resumes"
EMBEDDING_MODEL = "gemini-embedding-001"
TOP_K = 10


def retrieve_candidates(job_description: str, profession: str) -> str:
    """
    Embed the job description, query Qdrant with a profession filter,
    and return the top K candidate profiles as a formatted string.
    This runs BEFORE the crew starts — the crew never touches Qdrant directly.
    """
    result = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=job_description,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    query_vector = result.embeddings[0].values

    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        using="text_vectors",
        limit=TOP_K,
        with_payload=True,
    )

    if not hits:
        return "No candidates found in the database for this profession and job description."

    profiles = []
    for i, hit in enumerate(hits.points, 1):
        payload = hit.payload
        profiles.append(
            f"CANDIDATE {i} (similarity score: {hit.score:.3f})\n"
            f"Profession: {payload.get('profession', 'N/A')}\n"
            f"---\n"
            f"{payload.get('resume_content', 'No content available')}\n"
        )

    return "\n\n".join(profiles)

def build_tasks(job_description: str, profession: str) -> list:
    """
    Build the 4 tasks for the crew.
    Qdrant retrieval happens here (before agents run) and the results
    are injected directly into task 2's description as context.
    This means the crew processes all 10 candidates in parallel within
    each task — not one by one.
    """

    print(f"\nQuerying Qdrant for top {TOP_K} '{profession}' candidates...")
    retrieved_candidates = retrieve_candidates(job_description, profession)
    print(f"Retrieved {TOP_K} candidates. Handing off to agents...\n")

    # ── TASK 1 ──────────────────────────────────────────────────────────────
    task_analyze_job = Task(
        description=(
            f"Analyze the following job description thoroughly.\n\n"
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            f"Extract and structure the following:\n"
            f"1. Job title and department\n"
            f"2. Must-have technical skills (non-negotiable)\n"
            f"3. Preferred/nice-to-have skills\n"
            f"4. Required years of experience\n"
            f"5. Education requirements\n"
            f"6. Key responsibilities\n"
            f"7. Any soft skills or personality traits mentioned\n"
            f"8. Industry or domain knowledge required\n\n"
            f"Be precise. If something is ambiguous, flag it."
        ),
        expected_output=(
            "A structured requirements document with clearly labeled sections for: "
            "must-have skills, preferred skills, experience requirements, education, "
            "responsibilities, and soft skills. "
            "Format it clearly so other agents can reference it easily."
        ),
        agent=job_analyst,
    )

    # ── TASK 2 ──────────────────────────────────────────────────────────────
    task_present_candidates = Task(
        description=(
            f"The following {TOP_K} candidates have been retrieved from the resume database "
            f"based on semantic similarity to the job description. "
            f"Your job is to read each candidate's raw resume content and present it as a "
            f"clean, structured profile.\n\n"
            f"For each candidate extract:\n"
            f"- Full name (if available)\n"
            f"- Years of total experience\n"
            f"- Current or most recent job title\n"
            f"- Core technical skills (bullet list)\n"
            f"- Education (degree, institution)\n"
            f"- Notable achievements or projects\n"
            f"- Career highlights (2-3 sentences)\n\n"
            f"RETRIEVED CANDIDATES FROM DATABASE:\n\n"
            f"{retrieved_candidates}"
        ),
        expected_output=(
            f"A clean structured profile for each of the {TOP_K} candidates, "
            "clearly numbered (Candidate 1, Candidate 2, etc.) with all extracted "
            "fields filled in. If a field is not found in the resume, write 'Not specified'."
        ),
        agent=candidate_retriever,
        context=[task_analyze_job],
    )

    # ── TASK 3 ──────────────────────────────────────────────────────────────
    task_evaluate_candidates = Task(
        description=(
            f"You will receive the structured job requirements and the {TOP_K} candidate profiles. "
            f"Score EVERY candidate against the job requirements.\n\n"
            f"For each candidate produce a scorecard with:\n"
            f"- Overall fit score (0–10)\n"
            f"- Technical skills match score (0–10)\n"
            f"- Experience match score (0–10)\n"
            f"- Education match score (0–10)\n"
            f"- Top 3 strengths (specific, evidence-based)\n"
            f"- Top 2 gaps or concerns\n"
            f"- One-line verdict: STRONG FIT / GOOD FIT / PARTIAL FIT / WEAK FIT\n\n"
            f"Rules:\n"
            f"- Apply identical criteria to all candidates\n"
            f"- Every score must be backed by a specific reason\n"
            f"- Do not let candidate order influence your scores\n"
            f"- Process all {TOP_K} candidates — do not skip any"
        ),
        expected_output=(
            f"A complete scorecard for all {TOP_K} candidates. "
            "Each scorecard clearly shows all 4 scores, strengths, gaps, and verdict. "
            "End with a ranked list from highest to lowest overall fit score."
        ),
        agent=candidate_evaluator,
        context=[task_analyze_job, task_present_candidates],
    )

    # ── TASK 4 ──────────────────────────────────────────────────────────────
    task_write_report = Task(
        description=(
            "Write a professional hiring report based on the job requirements analysis "
            "and the candidate scorecards. The report will be read by a hiring committee.\n\n"
            "The report must contain these sections in this order:\n\n"
            "1. EXECUTIVE SUMMARY (3-4 sentences: what role, how many candidates reviewed, "
            "key finding, top recommendation)\n\n"
            "2. RANKED SHORTLIST TABLE (candidate name/ID, overall score, verdict — "
            "all 10 candidates in a clean table)\n\n"
            "3. TOP 3 CANDIDATES — DETAILED PROFILES (for the top 3 scorers only: "
            "full scorecard, why they stand out, suggested interview focus areas)\n\n"
            "4. CANDIDATES TO CONSIDER (candidates ranked 4–6 who are worth keeping "
            "in the pipeline)\n\n"
            "5. NOT RECOMMENDED (brief note on why the bottom candidates don't fit — "
            "one sentence each)\n\n"
            "6. FINAL RECOMMENDATION (who to interview first and why — be decisive)\n\n"
            "Tone: professional, direct, no HR jargon. Write for a busy executive."
        ),
        expected_output=(
            "A complete, polished hiring report with all 6 sections. "
            "Should be readable in under 5 minutes. "
            "The final recommendation must be a clear, unambiguous statement "
            "naming the top candidate and the reason."
        ),
        agent=report_writer,
        context=[task_analyze_job, task_evaluate_candidates],
    )

    return [
        task_analyze_job,
        task_present_candidates,
        task_evaluate_candidates,
        task_write_report,
    ]
