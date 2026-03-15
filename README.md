# AI HR Resume Screener

**A multi-agent AI system that screens and ranks job candidates using RAG, vector search, and a sequential CrewAI pipeline.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [How It Works — Three Phases](#3-how-it-works--three-phases)
4. [Project Structure](#4-project-structure)
5. [Prerequisites](#5-prerequisites)
6. [Installation](#6-installation)
7. [Configuration](#7-configuration)
8. [Running the System](#8-running-the-system)
9. [Module Reference](#9-module-reference)
   - [ingest.py](#91-ingestpy)
   - [llm.py](#92-llmpy)
   - [agents.py](#93-agentspy)
   - [tasks.py](#94-taskspy)
   - [main.py](#95-mainpy)
10. [The Agent Pipeline](#10-the-agent-pipeline)
11. [RAG & Qdrant — How Retrieval Works](#11-rag--qdrant--how-retrieval-works)
12. [LLM Strategy — Primary & Fallback](#12-llm-strategy--primary--fallback)
13. [Output — The Hiring Report](#13-output--the-hiring-report)
14. [Troubleshooting](#14-troubleshooting)
15. [Design Decisions & Rationale](#15-design-decisions--rationale)

---

## 1. Project Overview

The AI HR Resume Screener is a locally-runnable applied AI system that automates the first stage of the hiring pipeline: screening a large pool of resumes and producing a ranked shortlist with a written recommendation.

Given a job description and a target career category, the system:

1. Queries a Qdrant vector database containing 2,500 embedded resumes
2. Retrieves the top 10 most semantically relevant candidates using metadata-filtered vector search
3. Passes those candidates through a four-agent CrewAI pipeline that analyzes, evaluates, and ranks them
4. Outputs a professional `hiring_report.md` ready for a hiring committee

The system was built as a live demo for the NationAI Applied AI Instructor role assessment. It demonstrates RAG, multi-agent orchestration, vector databases, semantic search, LLM fallback strategies, and responsible AI considerations — all in a single cohesive application.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Offline Ingestion (runs once)                            │
│                                                                     │
│  2,500 PDFs  ──►  pdfplumber  ──►  Gemini Embed 001  ──►  Qdrant   │
│  (24 career dirs)   (extract)     (3072-dim vector)   (upsert +    │
│                                                         payload)    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    stored vectors persist in Qdrant
                                  │
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — Query & RAG (runs every time)                            │
│                                                                     │
│  HR Manager  ──►  Gemini Embed  ──►  Qdrant filtered search        │
│  (job desc +      (job desc →        (profession filter +           │
│   career)          vector)            cosine similarity)            │
│                                              │                      │
│                                        Top 10 candidates            │
└──────────────────────────────────────────────┼──────────────────────┘
                                               │
┌──────────────────────────────────────────────▼──────────────────────┐
│  PHASE 3 — CrewAI Agent Pipeline (sequential)                       │
│                                                                     │
│  Job Analyst ──► CV Retriever ──► Evaluator ──► Report Writer      │
│  (parse JD)     (clean profiles)  (score 0-10)  (hiring_report.md) │
│                                                                     │
│  LLM: Gemini 2.0 Flash  ──(fails 3×)──►  LLaMA 3.3 70B (Groq)     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. How It Works — Three Phases

### Phase 1 — Offline Ingestion

This phase runs **once** before any queries are made. It converts your resume library into a searchable vector database.

For each PDF in your dataset:
1. `pdfplumber` extracts raw text with layout-awareness (`x_tolerance=3, y_tolerance=3`)
2. The text is sent to **Gemini Embedding 001** which returns a 3072-dimensional float vector representing the semantic meaning of the resume
3. The vector is upserted into a Qdrant collection named `hr_resumes` along with metadata payload: the profession/career category, the raw resume text, and the file path

The ingestion pipeline includes rate limit handling (Gemini free tier: 100 req/min, 1,000 req/day), automatic retry on 429 errors, and a progress file so the process can be safely stopped and resumed across multiple days.

### Phase 2 — Query & RAG Retrieval

Each time a hiring manager wants to screen candidates:

1. The job description is embedded using the same Gemini model into a 3072-dim vector
2. Qdrant performs a **metadata-filtered vector search**: it first narrows the search space to only resumes in the specified career category (e.g. `profession="ENGINEERING"`), then finds the top 10 most similar vectors using cosine similarity
3. The 10 retrieved resume texts and their similarity scores are passed into the agent pipeline

This is the core RAG (Retrieval-Augmented Generation) step. The LLM agents never see the other 2,490 resumes — they only work with the pre-filtered shortlist.

### Phase 3 — CrewAI Agent Pipeline

Four specialized agents run sequentially, each receiving the outputs of all previous agents as context:

| Agent | Input | Output |
|---|---|---|
| Job Analyst | Raw job description | Structured requirements doc |
| CV Retriever | 10 raw resume texts + requirements | 10 clean structured profiles |
| Evaluator | Profiles + requirements | 10 scorecards (0–10) with verdicts |
| Report Writer | All scorecards | `hiring_report.md` |

The final output is a complete hiring report saved to disk.

---

## 4. Project Structure

```
project/
│
├── ingest.py                  # Phase 1: PDF extraction + Qdrant ingestion
├── llm.py                     # LLM configuration: primary + fallback logic
├── agents.py                  # 4 CrewAI agent definitions
├── tasks.py                   # 4 task definitions + Qdrant retrieval function
├── main.py                    # Entry point: configure job + run the crew
│
├── data/
│   └── data/
│       ├── ENGINEERING/       # One directory per career category
│       │   ├── 10001.pdf
│       │   └── ...
│       ├── FINANCE/
│       ├── FITNESS/
│       └── ...                # 24 total career directories
│
├── ingestion_progress.json    # Auto-generated: tracks ingested files
├── failed_ingestions.json     # Auto-generated: logs failed ingestions
├── hiring_report.md           # Auto-generated: final output report
│
└── .env                       # API keys (never commit to version control)
```

---

## 5. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| Docker | Any recent | For running Qdrant locally |
| Qdrant | Latest | Runs as a Docker container |
| Gemini API key | — | Free tier sufficient for testing |
| Groq API key | — | Free at console.groq.com |

---

## 6. Installation

### Step 1 — Clone and set up Python environment

```bash
git clone <your-repo>
cd hr-resume-screener
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Step 2 — Install dependencies

```bash
pip install crewai qdrant-client pdfplumber google-genai python-dotenv
```

### Step 3 — Start Qdrant with Docker

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

Qdrant will be available at `http://localhost:6333`. You can verify it's running by visiting `http://localhost:6333/dashboard` in your browser.

### Step 4 — Set up your resume dataset

Place your PDF resumes in the following structure:

```
data/
└── data/
    ├── ENGINEERING/
    │   ├── 10001.pdf
    │   └── 10002.pdf
    ├── FINANCE/
    └── FITNESS/
```

The directory names become the `profession` field stored in Qdrant. Use the exact same casing when querying (see [Configuration](#7-configuration)).

---

## 7. Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_key_here    # optional
COHERE_API_KEY=your_cohere_key_here            # optional
```

**Getting API keys:**
- **Gemini**: [aistudio.google.com](https://aistudio.google.com) → Get API key (free tier)
- **Groq**: [console.groq.com](https://console.groq.com) → Create API key (free tier)

In `main.py`, configure the job inputs:

```python
JOB_DESCRIPTION = """
We are looking for a Senior Data Engineer...
"""

PROFESSION = "ENGINEERING"   # Must exactly match a directory name in your dataset
```

> **Important:** `PROFESSION` is case-sensitive and must match exactly how the directory was named when ingestion ran. Run the snippet below to check what values are stored in Qdrant:
> ```python
> from qdrant_client import QdrantClient
> client = QdrantClient("http://localhost:6333")
> result = client.scroll("hr_resumes", limit=5, with_payload=True)
> print([p.payload["profession"] for p in result[0]])
> ```

---

## 8. Running the System

### Step 1 — Run ingestion (once)

```bash
python ingest.py
```

Expected output:
```
Collection 'hr_resumes' ready.
--------------------------------------------------
Progress file found: 0 resumes already ingested, skipping them.
--------------------------------------------------
Ingesting 'ENGINEERING' profession (98 resumes):
    - [1/2500] 10001.pdf ingested successfully.
    - [2/2500] 10002.pdf ingested successfully.
    ...
```

Ingestion respects the free tier rate limits automatically. If you stop it mid-way, simply re-run — it will skip already-processed files using `ingestion_progress.json`.

With 2,500 resumes and a 5-second delay between calls, full ingestion takes approximately **3.5 hours** of active processing time spread across 3 days (1,000 requests/day limit on free tier).

### Step 2 — Run the screener

```bash
python main.py
```

Expected output:
```
============================================================
  AI HR Resume Screener
  Powered by CrewAI + Qdrant RAG
============================================================
Checking primary LLM (Gemini)...
Gemini 2.0 Flash is available. Using as primary LLM.

Querying Qdrant for top 10 'ENGINEERING' candidates...
Retrieved 10 candidates. Handing off to agents...

[CrewAI verbose logs — agent thinking and actions]

============================================================
  FINAL HIRING REPORT
============================================================
[Report content printed here]

Report saved to hiring_report.md
```

Total runtime is typically **3–8 minutes** depending on the LLM and the length of the resumes.

---

## 9. Module Reference

### 9.1 `ingest.py`

Handles the complete offline ingestion pipeline: PDF text extraction, embedding generation, and Qdrant storage.

#### Key constants

| Constant | Default | Purpose |
|---|---|---|
| `DELAY_BETWEEN_REQUESTS` | `5` seconds | Pause between embedding API calls to respect rate limits |
| `MAX_RETRIES` | `5` | Maximum retry attempts on API failure |
| `RETRY_WAIT` | `60` seconds | Wait time when a rate limit error is hit |
| `PROGRESS_FILE` | `ingestion_progress.json` | Tracks which files have already been ingested |

#### Key functions

**`collection_creation()`**
Creates the Qdrant collection with a named vector config. Skips creation if the collection already exists, making it safe to call on every run.

```python
vectors_config={
    "text_vectors": models.VectorParams(
        size=3072,                      # Gemini embedding dimension
        distance=models.Distance.COSINE # Best for semantic text similarity
    )
}
```

**`get_embedding_with_retry(text: str) -> list`**
Calls the Gemini Embedding API and returns a 1D list of 3072 floats. Implements exponential-like retry logic:
- Sleeps `DELAY_BETWEEN_REQUESTS` seconds after every successful call
- On 429/quota/rate errors: waits `RETRY_WAIT` seconds and retries
- On other errors: also waits and retries (network blips, temporary outages)
- Raises `RuntimeError` after `MAX_RETRIES` consecutive failures

**`qdrant_upsert(id, text, file_path, profession)`**
Stores a single resume in Qdrant. The vector is stored under the named key `"text_vectors"` (required because the collection uses named vectors). The payload includes:

```python
payload={
    "file_path": file_path,        # Original PDF location on disk
    "profession": profession,       # Career category — used as filter key
    "resume_content": text          # Full extracted text — passed to agents
}
```

**`extract_resume_text(pdf_path: str) -> str`**
Extracts text from a PDF using `pdfplumber`. Uses `x_tolerance=3, y_tolerance=3` for better handling of multi-column and formatted layouts. Returns an empty string if extraction fails — this is caught by the caller and logged as a warning.

**`main()`**
Iterates over all career directories and resumes. For each PDF:
1. Skips if already in progress file
2. Skips if not a `.pdf` file
3. Extracts text; skips with warning if empty
4. Generates integer ID from filename (falls back to UUID if filename is non-numeric)
5. Upserts to Qdrant
6. Saves progress immediately after each successful upsert

---

### 9.2 `llm.py`

Manages LLM configuration and the primary-to-fallback resolution logic. All agents import the resolved LLM from this module.

#### Configured models

| Variable | Model | Provider | Role |
|---|---|---|---|
| `gemini_llm` | `gemini/gemini-3.1-flash-lite-preview` | Google | Primary |
| `groq_llm` | `groq/llama-3.3-70b-versatile` | Groq | Fallback (open source) |
| `cohere_llm` | `cohere/command-a-03-2025` | Cohere | Optional alternative |
| `hunter_llm` | `openrouter/hunter-alpha` | OpenRouter | Optional alternative |

All LLMs are configured via CrewAI's `LLM` class, which uses LiteLLM under the hood. This means model strings follow the `provider/model-name` format.

#### `get_llm_with_fallback(max_retries, retry_wait) -> LLM`

Performs a lightweight connectivity probe on the primary LLM before the crew starts. Returns the primary if available, falls back to Groq if the primary fails `max_retries` times.

This check happens **at import time** in `agents.py`, meaning the LLM is resolved once and shared across all four agents. There is no mid-run switching — the choice is made before the crew begins.

> **Current state:** The function currently returns `hunter_llm` directly (early return for testing). Remove the `return hunter_llm` line on line 2 of the function body to restore the full fallback logic.

---

### 9.3 `agents.py`

Defines the four CrewAI agents. Each agent has:
- **role**: a job title that anchors the agent's persona
- **goal**: a precise statement of what the agent must produce
- **backstory**: narrative context that shapes the agent's tone and priorities
- **`allow_delegation=False`**: prevents agents from handing tasks to each other (keeps the pipeline deterministic)
- **`llm=llm`**: the resolved LLM from `llm.py`, shared by all agents

#### Agent 1 — `job_analyst`

| Field | Value |
|---|---|
| Role | Senior Job Requirements Analyst |
| Input | Raw job description text |
| Output | Structured requirements document |
| Key behavior | Distinguishes must-haves from nice-to-haves; flags ambiguities |

#### Agent 2 — `candidate_retriever`

| Field | Value |
|---|---|
| Role | Talent Database Specialist |
| Input | 10 raw resume texts (pre-retrieved from Qdrant) + requirements doc |
| Output | 10 clean, structured candidate profiles |
| Key behavior | Extracts name, title, skills, years of experience, education, achievements. Does not score or judge. |

#### Agent 3 — `candidate_evaluator`

| Field | Value |
|---|---|
| Role | Objective Candidate Evaluator |
| Input | 10 structured profiles + requirements doc |
| Output | 10 scorecards with numerical scores and verdicts |
| Key behavior | Applies consistent rubric to all candidates; scores are evidence-backed; verdict is one of STRONG FIT / GOOD FIT / PARTIAL FIT / WEAK FIT |

#### Agent 4 — `report_writer`

| Field | Value |
|---|---|
| Role | Executive Talent Report Writer |
| Input | All scorecards + requirements doc |
| Output | `hiring_report.md` |
| Key behavior | Leads with recommendation; produces scannable executive format; avoids HR jargon |

---

### 9.4 `tasks.py`

Defines the four tasks and the Qdrant retrieval function. This is the only module that interacts with Qdrant at query time.

#### `retrieve_candidates(job_description, profession) -> str`

Embeds the job description using the Gemini embedding model, then queries Qdrant with a `profession` payload filter and a cosine similarity search:

```python
hits = qdrant.search(
    collection_name=COLLECTION,
    query_vector={"name": "text_vectors", "vector": query_vector},
    query_filter=Filter(
        must=[FieldCondition(key="profession", match=MatchValue(value=profession))]
    ),
    limit=TOP_K,           # Default: 10
    with_payload=True,
)
```

Returns a formatted multi-line string of all retrieved candidate profiles, ready to be injected into Task 2's description.

#### `build_tasks(job_description, profession) -> list[Task]`

The main function called by `main.py`. It:
1. Calls `retrieve_candidates()` to fetch the top 10 from Qdrant
2. Builds all four `Task` objects with the retrieved data already baked into the descriptions
3. Returns the task list in sequential order

Tasks are connected via the `context` parameter — each task receives the outputs of the tasks listed in its `context`:

```python
task_evaluate = Task(
    ...
    context=[task_analyze_job, task_present_candidates],  # receives both prior outputs
)
```

#### Top K configuration

```python
TOP_K = 10  # number of candidates retrieved from Qdrant
```

Reduce to `5` if you encounter LLM context window errors (especially on Groq's free tier, which has a 32k token limit for LLaMA 3.3 70B).

---

### 9.5 `main.py`

The entry point. Configure your job inputs here and run the crew.

#### Inputs

```python
JOB_DESCRIPTION = """
... your job description text ...
"""

PROFESSION = "ENGINEERING"    # must match profession payload in Qdrant exactly
```

#### What it does

1. Prints a startup banner
2. Calls `build_tasks()` which triggers Qdrant retrieval
3. Assembles the `Crew` with all four agents and tasks
4. Runs the crew with `Process.sequential`
5. Prints the final report to stdout
6. Saves the report to `hiring_report.md`

---

## 10. The Agent Pipeline

The pipeline is sequential: each agent completes its task fully before the next agent begins. The context chain ensures every agent has access to all prior outputs.

```
Job Description
      │
      ▼
┌─────────────────────────────┐
│  Task 1: Job Analyst        │  ← gets: job description
│  Produces: requirements doc │
└──────────────┬──────────────┘
               │ context
               ▼
┌─────────────────────────────┐
│  Task 2: CV Retriever       │  ← gets: requirements doc + 10 raw resumes
│  Produces: 10 profiles      │
└──────────────┬──────────────┘
               │ context
               ▼
┌─────────────────────────────┐
│  Task 3: Evaluator          │  ← gets: requirements doc + 10 profiles
│  Produces: 10 scorecards    │
└──────────────┬──────────────┘
               │ context
               ▼
┌─────────────────────────────┐
│  Task 4: Report Writer      │  ← gets: requirements doc + 10 scorecards
│  Produces: hiring_report.md │
└─────────────────────────────┘
```

All 10 candidates are processed within each task in a single LLM call. The crew does not loop over candidates one by one — this keeps the pipeline fast and allows the evaluator to compare candidates relatively rather than in isolation.

---

## 11. RAG & Qdrant — How Retrieval Works

### Why a vector database?

Traditional keyword search cannot match "5 years building ETL pipelines" with a job description that says "data engineering experience." Vector databases solve this by converting text into numerical representations (embeddings) where semantically similar content is close together in mathematical space.

### Single vector per resume

Each resume is stored as one vector, not chunked into sections. This is the correct approach for candidate matching because:

- The unit of retrieval is a **person**, not a fragment of a document
- A resume is short enough (600–800 tokens) to fit comfortably within the embedding model's input limit
- Cross-section relationships (e.g. a candidate whose skills and experience together match the role) are preserved in a single whole-document vector

### Metadata filtering

When the HR manager specifies a career category, Qdrant applies the filter **before** the vector search. This means:

- For a dataset with 2,500 resumes across 24 careers (~104 resumes per category), the search scans ~100 vectors instead of 2,500
- Results are guaranteed to be from the correct career domain
- The system scales cleanly to 10,000+ resumes without degrading result quality

### Cosine similarity

The collection uses cosine distance, which measures the angle between two vectors rather than their magnitude. This is optimal for text embeddings because it captures semantic direction (meaning) rather than length (verbosity). A long resume and a short resume with similar content will score similarly.

---

## 12. LLM Strategy — Primary & Fallback

### Why two LLMs?

Building demos on a single API creates a single point of failure. Rate limits, temporary outages, and quota exhaustion are common during testing. A fallback LLM eliminates this risk.

### Resolution logic

```
Startup
   │
   ├─► Probe Gemini (attempt 1 of 3)
   │     Success? ──► Use Gemini for all agents
   │     Failure? ──► Wait 15s, retry
   │
   ├─► Probe Gemini (attempt 2 of 3)
   │     Success? ──► Use Gemini for all agents
   │     Failure? ──► Wait 15s, retry
   │
   ├─► Probe Gemini (attempt 3 of 3)
   │     Success? ──► Use Gemini for all agents
   │     Failure? ──► Switch to fallback
   │
   └─► Use LLaMA 3.3 70B (Groq)
```

The LLM is resolved **once at startup**, before the crew runs. All four agents share the same resolved instance. There is no mid-run switching.

### Model comparison

| | Gemini 2.0 Flash | LLaMA 3.3 70B (Groq) |
|---|---|---|
| Type | Proprietary | Open source |
| Provider | Google | Meta (via Groq) |
| Speed | Fast | Very fast (Groq inference) |
| Context window | 1M tokens | 128k tokens |
| Free tier | Yes | Yes |
| Best for | Long context, structured output | Strong reasoning, open source demos |

### Groq context limit note

Groq's LLaMA 3.3 70B has a 128k context window, but the free tier request limit is lower. If you pass 10 full resumes to the agents and hit a context error, reduce `TOP_K` in `tasks.py` from `10` to `5`.

---

## 13. Output — The Hiring Report

The final output is a markdown file (`hiring_report.md`) with six sections:

```
# Hiring Report — [Role] — [Date]

## 1. Executive Summary
3–4 sentences: what role, how many reviewed, key finding, top recommendation.

## 2. Ranked Shortlist Table
| Rank | Candidate | Overall Score | Verdict |
|------|-----------|---------------|---------|
| 1    | ...       | 8.5/10        | STRONG FIT |
...

## 3. Top 3 Candidates — Detailed Profiles
Full scorecard for candidates ranked 1–3.
Suggested interview focus areas for each.

## 4. Candidates to Consider
Candidates ranked 4–6 worth keeping in the pipeline.

## 5. Not Recommended
One-sentence explanation for why each of the bottom candidates does not fit.

## 6. Final Recommendation
A clear, unambiguous hiring recommendation naming the top candidate and the reason.
```

---

## 14. Troubleshooting

### Qdrant connection refused

```
httpx.ConnectError: [Errno 111] Connection refused
```

Qdrant is not running. Start it:
```bash
docker start qdrant
# or if first time:
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### Collection already exists error during ingestion

`ingest.py` checks for existing collections before creating. If you see an unexpected error here, check that your Qdrant container is the correct version and that the collection schema matches what the code expects.

### Empty profession filter returns no results

```
No candidates found in the database for this profession and job description.
```

The `PROFESSION` value in `main.py` does not match what is stored in Qdrant. Check stored values:
```python
from qdrant_client import QdrantClient
client = QdrantClient("http://localhost:6333")
result = client.scroll("hr_resumes", limit=5, with_payload=True)
print(set(p.payload["profession"] for p in result[0]))
```

### Gemini rate limit during ingestion

```
[Rate limit] Attempt 1/5 — waiting 60s before retry...
```

This is expected behavior, not an error. The retry logic will handle it automatically. If ingestion stops entirely after 5 retries, you have hit the daily quota (1,000 requests/day on free tier). Stop the script and re-run it the next day — `ingestion_progress.json` will resume from where you left off.

### LLM context window error

```
Error: context length exceeded
```

Reduce the number of retrieved candidates. In `tasks.py`:
```python
TOP_K = 5  # reduce from 10
```

### CrewAI agents not seeing each other's outputs

Ensure the `context` parameter in each task includes all prior tasks that agent needs:
```python
task_evaluate = Task(
    ...
    context=[task_analyze_job, task_present_candidates],  # both required
)
```

### Named vector error on upsert

```
VectorError: Vector 'text_vectors' not found
```

The collection was created with named vectors but the upsert is not using the vector name. Ensure the upsert uses:
```python
vector={"text_vectors": embeddings}   # dict, not a plain list
```

---

## 15. Design Decisions & Rationale

### Why CrewAI over a single LLM call?

A single LLM call with all instructions produces inconsistent results when the prompt is complex. Breaking the work into four focused agents — each with a clear, narrow responsibility — produces more reliable, auditable output. Each agent's output can be inspected independently, and if one step fails the others are unaffected.

### Why Qdrant over a simple list comparison?

With 2,500 resumes, sending all of them to an LLM is prohibitively expensive (token cost and latency). Qdrant allows the system to retrieve only the semantically relevant subset in milliseconds, with metadata filtering ensuring the results come from the correct career domain. This is the RAG pattern — retrieval before generation.

### Why a single vector per resume instead of chunking?

Chunking is appropriate when the unit of retrieval is a section of a document (e.g. retrieving a specific clause from a 50-page legal contract). Here, the unit of retrieval is a whole person. A resume is short enough to embed as a single document (600–900 tokens), and a whole-document vector preserves cross-section relationships (skills + experience together) that averaged chunk vectors would lose.

### Why metadata filtering instead of separate collections?

Separate Qdrant collections per career category would achieve the same filtering effect but would require managing 24 connection objects, separate upsert logic per collection, and complex queries if you ever want to search across categories. Payload metadata filtering achieves the same result — pre-filtering the search space before vector comparison — with a single collection and a one-line filter parameter.

### Why cosine distance instead of dot product or Euclidean?

Cosine similarity measures the angle between vectors, which captures semantic direction regardless of vector magnitude. For text embeddings, a long detailed resume and a short resume covering the same skills will produce vectors of different magnitudes but similar directions. Cosine distance treats them as equally similar to a matching query. Euclidean distance would penalize the shorter resume unfairly.

### Why Gemini embeddings for both ingestion and query?

The embedding model used to encode resumes must be identical to the model used to encode the query at search time. Using different models would produce vectors in different mathematical spaces, making similarity scores meaningless. Both ingestion and query use `gemini-embedding-001` with `task_type="SEMANTIC_SIMILARITY"`.

---

*Built for NationAI · Applied AI Instructor Demo · March 2026*
