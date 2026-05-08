from crewai import Task
from core.agents import job_analyst, candidate_retriever, candidate_evaluator, report_writer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from google import genai
from google.genai import types
from core.settings_config import settings

qdrant_config = settings.qdrant_host + ":" + str(settings.qdrant_port)
qdrant = QdrantClient(qdrant_config, timeout=settings.qdrant_timeout)
gemini_client = genai.Client()

REPORT_SECTIONS = [
    "ROLE REQUIREMENTS RECAP",
    "EXECUTIVE SUMMARY",
    "CANDIDATE SNAPSHOTS",
    "RANKED SHORTLIST TABLE",
    "TOP 3 CANDIDATES — DETAILED PROFILES",
    "PIPELINE CANDIDATES",
    "NOT RECOMMENDED",
    "RISK & DATA QUALITY NOTES",
    "NEXT STEPS",
    "FINAL RECOMMENDATION",
]

def retrieve_candidates(job_description: str, profession: str) -> str:
    """
    Embed the job description, query Qdrant with a profession filter,
    and return the top K candidate profiles as a formatted string.
    This runs BEFORE the crew starts — the crew never touches Qdrant directly.
    """
    result = gemini_client.models.embed_content(
        model=settings.embedding_model,
        contents=job_description,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    query_vector = result.embeddings[0].values

    hits = qdrant.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        using="text_vectors",
        limit=settings.top_k,
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
    This means the crew processes all 10 candidates in one pass per task.
    """
 
    print(f"\nQuerying Qdrant for top {settings.top_k} '{profession}' candidates...")
    retrieved_candidates = retrieve_candidates(job_description, profession)
    print(f"Retrieved {settings.top_k} candidates. Handing off to agents...\n")
 
    # ── TASK 1: Job Analysis ─────────────────────────────────────────────────
    task_analyze_job = Task(
        description=(
            f"Analyze the following job description thoroughly.\n\n"
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            f"Extract and structure the following:\n"
            f"1. Job title and department\n"
            f"2. Must-have technical skills — label each with priority weight: "
            f"   HIGH (role cannot function without it) / MEDIUM / LOW\n"
            f"3. Preferred / nice-to-have skills (with weight)\n"
            f"4. Required years of experience (overall and domain-specific)\n"
            f"5. Education requirements (degree, field, hard requirement vs preferred)\n"
            f"6. Key responsibilities ranked by apparent importance\n"
            f"7. Soft skills or personality traits mentioned or implied\n"
            f"8. Industry or domain knowledge required\n"
            f"9. Any ambiguous or buzzword-heavy language — flag it and state your "
            f"   concrete interpretation\n"
            f"10. Realistic hiring context note: typical notice period for this role, "
            f"    seniority level, and any urgency signals in the description\n\n"
            f"Be precise. If something is ambiguous, flag it and explain your interpretation."
        ),
        expected_output=(
            "A structured requirements document with clearly labeled sections covering "
            "all 10 extraction points above. "
            "Each skill or requirement must carry a priority weight (HIGH / MEDIUM / LOW). "
            "The document must be unambiguous and directly usable as a scoring rubric "
            "by the candidate evaluator."
        ),
        agent=job_analyst,
    )
 
    # ── TASK 2: Candidate Presentation ──────────────────────────────────────
    task_present_candidates = Task(
        description=(
            f"The following {settings.top_k} candidates have been retrieved from the resume database "
            f"based on semantic similarity to the job description. "
            f"Read each candidate's raw resume and produce a clean, structured profile.\n\n"
            f"For each candidate extract:\n"
            f"- Full name (or 'Not specified')\n"
            f"- Current or most recent job title\n"
            f"- Total years of professional experience\n"
            f"- Core technical skills (bullet list, be specific — no generic terms)\n"
            f"- Education (degree, field, institution, year if available)\n"
            f"- Notable achievements or measurable results\n"
            f"- Career highlights (2-3 sentences)\n"
            f"- Employment timeline: flag any gaps > 6 months or very short tenures "
            f"  (< 1 year) — do not editorialize, just note them factually\n"
            f"- Data quality flag: note any missing, inconsistent, or implausible "
            f"  information so the evaluator can account for uncertainty\n\n"
            f"RETRIEVED CANDIDATES FROM DATABASE:\n\n"
            f"{retrieved_candidates}"
        ),
        expected_output=(
            f"A clean structured profile for each of the {settings.top_k} candidates, "
            "clearly numbered (Candidate 1, Candidate 2, etc.). "
            "Every field must be filled in or explicitly marked 'Not specified'. "
            "Each profile ends with a data quality flag (Clean / Minor gaps / Significant gaps) "
            "with a one-sentence explanation if not Clean."
        ),
        agent=candidate_retriever,
        context=[task_analyze_job],
    )
 
    # ── TASK 3: Candidate Evaluation ─────────────────────────────────────────
    task_evaluate_candidates = Task(
        description=(
            f"You will receive the structured job requirements (with priority weights) "
            f"and the {settings.top_k} clean candidate profiles. "
            f"Score EVERY candidate. Do not skip any.\n\n"
            f"For each candidate produce a scorecard:\n"
            f"- Overall weighted fit score (0–10): weight technical skills highest, "
            f"  then experience, then education, per the priority weights in the requirements\n"
            f"- Technical skills match score (0–10) with brief justification\n"
            f"- Experience match score (0–10) with brief justification\n"
            f"- Education match score (0–10) with brief justification\n"
            f"- Top 3 strengths — specific, evidence-based, tied to job requirements\n"
            f"- Top 2 gaps or risk flags — be honest; include employment gaps, "
            f"  missing critical skills, or data quality issues\n"
            f"- Compensation fit: note if experience level suggests likely mismatch "
            f"  with typical band for this role (if inferable)\n"
            f"- Verdict: STRONG FIT / GOOD FIT / PARTIAL FIT / WEAK FIT\n\n"
            f"Scoring rules:\n"
            f"- Apply identical criteria to all {settings.top_k} candidates\n"
            f"- Every score must cite specific evidence from the profile\n"
            f"- Do not let candidate order or name influence scores\n"
            f"- If a profile had data quality issues, note how that affected scoring confidence\n\n"
            f"End with a ranked list (highest to lowest overall score) and a one-paragraph "
            f"synthesis noting the overall pool quality and any patterns worth flagging "
            f"(e.g., 'most candidates lack X', 'two candidates have very similar profiles')."
        ),
        expected_output=(
            f"A complete scorecard for all {settings.top_k} candidates with all required fields. "
            "Followed by a ranked list from highest to lowest overall fit score. "
            "Followed by a pool quality synthesis paragraph."
        ),
        agent=candidate_evaluator,
        context=[task_analyze_job, task_present_candidates],
    )
 
    # ── TASK 4: Report Writing ────────────────────────────────────────────────
    section_list = "\n".join(
        f"   {i+1}. {s}" for i, s in enumerate(REPORT_SECTIONS)
    )
    task_write_report = Task(
        description=(
            "Write a professional, decision-ready hiring report based on the job "
            "requirements analysis and candidate scorecards. "
            "The audience is a busy hiring committee who will read this once and act on it.\n\n"
            f"The report must contain these {len(REPORT_SECTIONS)} sections in this order:\n\n"
            f"{section_list}\n\n"
            "Section guidance:\n\n"
            "1. ROLE REQUIREMENTS RECAP — 3-5 bullet points summarising the non-negotiables. "
            "   Gives the committee a shared reference without reading the full JD.\n\n"
            "2. EXECUTIVE SUMMARY — 3-4 sentences: overall pool quality, top finding, "
            "   and the single top recommendation. Lead with the recommendation.\n\n"
            "3. CANDIDATE SNAPSHOTS — one short paragraph per candidate (all 10): "
            "   role, experience level, one key strength, one key gap. "
            "   This is the quick-scan layer before the detailed sections.\n\n"
            "4. RANKED SHORTLIST TABLE — all 10 candidates in a clean table with columns: "
            "   Rank | Name/ID | Overall Score | Verdict | One-Line Note.\n\n"
            "5. TOP 3 CANDIDATES — DETAILED PROFILES — for the top 3 scorers only: "
            "   full scorecard, why they stand out relative to the role, "
            "   suggested interview focus areas (2-3 specific questions or themes per candidate), "
            "   and recommended interview panel (e.g., hiring manager + technical lead).\n\n"
            "6. PIPELINE CANDIDATES — candidates worth keeping warm (typically GOOD FIT / "
            "   PARTIAL FIT): one paragraph each covering strengths, gaps, and "
            "   what would need to be true to move them forward.\n\n"
            "7. NOT RECOMMENDED — one sentence per candidate explaining the disqualifying factor. "
            "   Omit this section entirely if no candidates fall here.\n\n"
            "8. RISK & DATA QUALITY NOTES — flag any scoring uncertainty caused by "
            "   incomplete profiles, employment gaps, or compensation mismatches. "
            "   Keep it factual and brief.\n\n"
            "9. NEXT STEPS — a concrete checklist: who reaches out to the top candidate, "
            "   who schedules, suggested decision deadline, and whether the pipeline should "
            "   be kept open or closed.\n\n"
            "10. FINAL RECOMMENDATION — one decisive paragraph: who to interview first, "
            "    why, and what success looks like. No hedging.\n\n"
            "Tone: professional, direct, no HR jargon, no filler sentences. "
            "If a section has nothing meaningful to say, write 'N/A — [reason]' rather than padding."
        ),
        expected_output=(
            f"A complete, polished hiring report with all {len(REPORT_SECTIONS)} sections "
            "in the specified order. "
            "Readable in under 5 minutes. "
            "The FINAL RECOMMENDATION must be a clear, unambiguous statement "
            "naming the top candidate and the decisive reason. "
            "The report must be self-contained — a reader with no prior context "
            "should be able to act on it immediately."
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