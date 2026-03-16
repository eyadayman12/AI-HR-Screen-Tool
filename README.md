# AI HR Resume Screener — Complete Documentation

> **Version:** 1.0 · **Stack:** Python · CrewAI · Qdrant · Gemini · Streamlit  
> **Built for:** NationAI · Applied AI Instructor Demo · March 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Who This Is For](#2-who-this-is-for)
3. [What Problem It Solves](#3-what-problem-it-solves)
4. [System Architecture](#4-system-architecture)
5. [How It Works — Three Phases](#5-how-it-works--three-phases)
6. [Project File Structure](#6-project-file-structure)
7. [Module Reference](#7-module-reference)
   - [llm.py — LLM Configuration & Fallback](#71-llmpy--llm-configuration--fallback)
   - [agents.py — The Four AI Agents](#72-agentspy--the-four-ai-agents)
   - [tasks.py — Task Definitions & Qdrant Retrieval](#73-taskspy--task-definitions--qdrant-retrieval)
   - [ingest.py — Resume Ingestion Pipeline](#74-ingestpy--resume-ingestion-pipeline)
   - [main.py — CLI Entry Point](#75-mainpy--cli-entry-point)
   - [app.py — Streamlit UI](#76-apppy--streamlit-ui)
8. [The Agent Pipeline — Deep Dive](#8-the-agent-pipeline--deep-dive)
9. [RAG & Vector Search — How Retrieval Works](#9-rag--vector-search--how-retrieval-works)
10. [LLM Strategy — Resilient Multi-Provider Fallback](#10-llm-strategy--resilient-multi-provider-fallback)
11. [The Streamlit UI — Feature Guide](#11-the-streamlit-ui--feature-guide)
12. [Prerequisites & Installation](#12-prerequisites--installation)
13. [Configuration — Environment Variables](#13-configuration--environment-variables)
14. [Running the System](#14-running-the-system)
15. [Output — The Hiring Report](#15-output--the-hiring-report)
16. [Known Issues & Fixes](#16-known-issues--fixes)
17. [Design Decisions & Rationale](#17-design-decisions--rationale)
18. [Responsible AI Considerations](#18-responsible-ai-considerations)
19. [Glossary](#19-glossary)

---

## 1. Project Overview

The **AI HR Resume Screener** is a locally-runnable, applied AI system that automates the first stage of the hiring pipeline: screening a large pool of resumes and producing a ranked shortlist with a written executive recommendation — ready for a hiring committee.

Given a job description and a target career category, the system:

1. Queries a **Qdrant vector database** containing 2,500 embedded resumes using semantic similarity search
2. Retrieves the **top 10 most relevant candidates** using metadata-filtered vector search
3. Passes those candidates through a **four-agent CrewAI pipeline** that analyzes, evaluates, and ranks them
4. Outputs a polished **hiring report** viewable in the Streamlit UI and downloadable as Markdown

The system demonstrates five advanced AI engineering concepts working together in a single cohesive application: **RAG (Retrieval-Augmented Generation)**, **multi-agent orchestration**, **vector databases**, **semantic search**, and **LLM fallback strategies**.

---

## 2. Who This Is For

| Audience | How They Use It |
|---|---|
| **HR Managers / Recruiters** | Paste a job description, pick a talent category, click Run — receive a decision-ready report in minutes |
| **Hiring Committees** | Read the final report to shortlist candidates for interviews |
| **AI / Engineering Teams** | Study the architecture as a reference implementation of production-grade RAG + multi-agent systems |
| **Stakeholders / Executives** | Use the Streamlit demo to evaluate the system's capabilities and ROI potential |

---

## 3. What Problem It Solves

**The traditional resume screening problem:**

A typical job posting receives 100–500 applications. A human recruiter spends 6–8 seconds per resume on the first pass. This process is:

- **Slow** — reviewing 200 resumes takes 4–6 hours of focused work
- **Inconsistent** — fatigue, order effects, and unconscious bias affect decisions
- **Opaque** — ranking decisions are rarely documented or justifiable
- **Expensive** — recruiter time is a significant cost in the hiring pipeline

**What this system does instead:**

- Retrieves only the semantically relevant candidates from a 2,500-resume database in milliseconds
- Applies identical, documented scoring criteria to every candidate
- Produces a structured, evidence-based scorecard for each person
- Delivers a complete hiring recommendation in a single, readable document
- Runs in minutes, not hours

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Offline Ingestion (runs once, up-front)                       │
│                                                                          │
│  2,500 PDFs  ──►  pdfplumber  ──►  Gemini Embed 001  ──►  Qdrant        │
│  (24 career        (extract         (3072-dim vector)    (upsert +       │
│   directories)      raw text)                             payload)       │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                    Vectors persist in Qdrant between runs
                                   │
┌──────────────────────────────────▼───────────────────────────────────────┐
│  PHASE 2 — Query & RAG Retrieval (runs on every screening request)       │
│                                                                          │
│  HR Manager  ──►  Gemini Embed  ──►  Qdrant filtered cosine search      │
│  (job desc +      (job desc →        (profession filter +                │
│   profession)      3072-dim vec)      top-10 by similarity)              │
│                                               │                          │
│                                         10 candidate profiles            │
└───────────────────────────────────────────────┼──────────────────────────┘
                                                │
┌───────────────────────────────────────────────▼──────────────────────────┐
│  PHASE 3 — CrewAI Agent Pipeline (sequential, context-chained)           │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐           │
│  │  Job Analyst    │─►│ CV Retriever    │─►│  Evaluator     │─► ──►     │
│  │  (parse JD)     │  │ (clean profiles)│  │  (score 0–10)  │           │
│  └─────────────────┘  └─────────────────┘  └────────────────┘           │
│                                                         │                │
│                                              ┌──────────▼──────────┐    │
│                                              │   Report Writer     │    │
│                                              │  (hiring_report.md) │    │
│                                              └─────────────────────┘    │
│                                                                          │
│  LLM: Gemini 3.1 Pro  ──(fails)──►  Cohere Command-A                   │
│                       ──(fails)──►  Hunter Alpha (OpenRouter)           │
│                       ──(fails)──►  LLaMA 3.3 70B (Groq)               │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼──────────────────────────────────────┐
│  STREAMLIT UI — app.py                                                   │
│                                                                          │
│  Left Panel: Config + Agent Status Cards + Progress Bar                 │
│  Right Panel: Live Log Feed | Candidate CV Cards | Final Report         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. How It Works — Three Phases

### Phase 1 — Offline Ingestion

This phase runs **once** to populate the vector database. It never needs to run again unless new resumes are added.

For each PDF resume in the dataset:

1. **Text extraction** — `pdfplumber` reads each page with layout-aware tolerances (`x_tolerance=3, y_tolerance=3`) to handle multi-column layouts and inconsistent spacing common in resumes
2. **Embedding** — the full resume text is sent to `gemini-embedding-001`, which returns a **3,072-dimensional float vector** encoding the semantic meaning of the entire document
3. **Storage** — the vector and its metadata (profession category, raw text, file path) are upserted into Qdrant's `hr_resumes` collection

The ingestion pipeline includes:
- Rate-limit handling (auto-retry with 60-second backoff on HTTP 429)
- A `ingestion_progress.json` checkpoint file so ingestion can be safely stopped and resumed across multiple sessions
- Failed-file logging to `failed_ingestions.json`

### Phase 2 — Query & RAG Retrieval

Each time a screening is requested:

1. The job description text is embedded using the same `gemini-embedding-001` model into a 3,072-dim vector
2. Qdrant performs a **cosine similarity search** filtered to the specified profession category
3. The **top 10 most semantically similar** resume vectors are returned along with their payloads (raw text + metadata)
4. The 10 candidate profiles are passed into the agent pipeline as structured context

This is the RAG step — the LLM agents never see the other 2,490 resumes. They only work with the pre-filtered, semantically-ranked shortlist.

### Phase 3 — CrewAI Agent Pipeline

Four specialized agents run sequentially. Each agent receives the complete outputs of all prior agents as context via the `context=` parameter on each `Task`.

| Step | Agent | Input | Output |
|---|---|---|---|
| 1 | Job Requirements Analyst | Raw job description | Structured requirements doc with priority weights |
| 2 | Talent Database Specialist | 10 raw CVs + requirements | 10 clean, structured candidate profiles |
| 3 | Objective Candidate Evaluator | Profiles + requirements | 10 scorecards (0–10) with verdicts and ranked list |
| 4 | Executive Report Writer | All scorecards + requirements | Complete hiring report |

---

## 6. Project File Structure

```
project/
│
├── app.py                     # Streamlit UI — primary demo interface
├── main.py                    # CLI entry point (non-UI run)
│
├── agents.py                  # 4 CrewAI agent definitions
├── tasks.py                   # 4 task definitions + Qdrant retrieval
├── llm.py                     # LLM config: 4 providers + ResilientLLM wrapper
├── ingest.py                  # Phase 1: PDF extraction + Qdrant ingestion
│
├── data/
│   └── data/
│       ├── ENGINEERING/       # One directory per career category (24 total)
│       │   ├── 10001.pdf
│       │   └── 10002.pdf
│       ├── FINANCE/
│       ├── FITNESS/
│       └── ...
│
├── ingestion_progress.json    # Auto-generated: tracks ingested file paths
├── failed_ingestions.json     # Auto-generated: logs ingestion failures
├── hiring_report.md           # Auto-generated: final output report (CLI mode)
│
└── .env                       # API keys — never commit to version control
```

---

## 7. Module Reference

### 7.1 `llm.py` — LLM Configuration & Fallback

This module defines all LLM providers and the `ResilientLLM` wrapper that handles mid-run failures transparently.

**Four providers are configured:**

| Provider | Model | Temperature | Use |
|---|---|---|---|
| Google Gemini | `gemini/gemini-3.1-pro-preview` | 0.2 | Primary |
| Cohere | `cohere/command-a-03-2025` | 0.0 | Fallback 1 |
| OpenRouter (Hunter) | `openrouter/hunter-alpha` | 0.0 | Fallback 2 |
| Groq (LLaMA) | `groq/llama-3.3-70b-versatile` | 0.2 | Fallback 3 |

**`ResilientLLM` class:**

A wrapper around any `crewai.LLM` instance that intercepts every `.call()` invocation. On failure (rate limit, provider error, or any exception), it cascades down the `FALLBACK_CHAIN` list and transparently switches providers without interrupting the crew's execution.

Key behaviors:
- Detects rate limits via `429`, `quota`, `rate limit`, `too many` keywords in error strings
- Detects provider errors via `400`, `bad request`, `stealth` keywords
- Waits 3 seconds between provider switches to avoid thundering-herd effects
- Raises `RuntimeError` only if all four providers are exhausted
- Exposes the active provider name via `__repr__` for debugging

**`get_llm_with_fallback()` function:**

Probes each provider in order with a minimal "reply with the single word: ok" test call at startup. Returns a `ResilientLLM` starting at the first provider that responds successfully. This ensures the crew never starts with a known-dead provider.

> **Note:** In the current version of `llm.py`, the probe loop is bypassed and `gemini_llm` is returned directly. This is intentional — it skips the startup delay when Gemini is reliably available.

---

### 7.2 `agents.py` — The Four AI Agents

Defines four `crewai.Agent` instances. All share the same LLM instance resolved once at import time via `get_llm_with_fallback()`. All have `allow_delegation=False` — no agent can hand tasks to another, ensuring clean sequential execution.

**Agent 1 — `job_analyst` (Senior Job Requirements Analyst)**

Receives the raw job description and produces a structured requirements document. Its core value is not just extraction but *interpretation* — it flags vague or buzzword-heavy language (e.g., "strong communication skills") and translates it into concrete, scorable criteria. It assigns `HIGH / MEDIUM / LOW` priority weights to every requirement, which become the weighting rubric for the evaluator downstream.

**Agent 2 — `candidate_retriever` (Talent Database Specialist)**

Receives the 10 raw resume texts and the requirements doc. Its job is data cleaning and structuring — not evaluation. It produces a consistent profile format for each candidate, explicitly flags data quality issues (employment gaps >6 months, inconsistent dates, missing fields), and marks each profile with a quality flag: `Clean / Minor gaps / Significant gaps`. It has zero interest in judging fit.

**Agent 3 — `candidate_evaluator` (Objective Candidate Evaluator)**

Receives the clean profiles and the weighted requirements. Scores every candidate 0–10 across four dimensions: overall weighted fit, technical skills, experience, and education. Every score must cite specific evidence from the profile — no gut feelings, no order bias. Produces a ranked list and a synthesis paragraph noting pool-wide patterns (e.g., "most candidates lack dbt experience").

**Agent 4 — `report_writer` (Executive Talent Report Writer)**

Receives all prior outputs and synthesizes them into a single, decision-ready document. Backstory is modeled on a former Chief People Officer — the writing style is direct, zero HR jargon, and leads with the recommendation rather than burying it. The report is designed to be readable in under 5 minutes.

---

### 7.3 `tasks.py` — Task Definitions & Qdrant Retrieval

This module has two responsibilities: executing the Qdrant vector search and building the four `crewai.Task` objects.

**`retrieve_candidates(job_description, profession)`**

Embeds the job description using `gemini-embedding-001` and queries Qdrant. Returns a formatted string of 10 candidate profiles with similarity scores. Note: the profession filter in `tasks.py` does **not** pass a Qdrant metadata filter — it queries the full collection and returns the top-K by similarity. If profession-specific filtering is needed, a `Filter(must=[FieldCondition(...)])` should be added to the `query_points` call.

**`build_tasks(job_description, profession)`**

Calls `retrieve_candidates` once before the crew starts, then injects the result directly into Task 2's description string. This means Qdrant retrieval happens outside the agent pipeline — agents never make API calls to Qdrant directly.

Task context chaining:
- Task 1 has no context dependencies
- Task 2 receives `context=[task_analyze_job]`
- Task 3 receives `context=[task_analyze_job, task_present_candidates]`
- Task 4 receives `context=[task_analyze_job, task_evaluate_candidates]`

The `REPORT_SECTIONS` list defines the exact structure the report writer must follow, injected as an ordered list into Task 4's description.

---

### 7.4 `ingest.py` — Resume Ingestion Pipeline

Handles Phase 1 — the one-time population of the Qdrant vector database.

**Key functions:**

- `collection_creation()` — creates the `hr_resumes` collection with named vectors (`text_vectors`, 3072 dimensions, cosine distance) if it doesn't already exist
- `extract_resume_text(pdf_path)` — uses `pdfplumber` to extract text from all pages with layout-aware tolerances
- `get_embedding_with_retry(text)` — calls Gemini Embedding API with retry logic; handles 429 rate limits with 60-second backoff, up to 5 attempts
- `qdrant_upsert(id, text, file_path, profession)` — stores the embedding vector and metadata payload in Qdrant
- `load_progress() / save_progress(ingested)` — checkpointing to `ingestion_progress.json` so long ingestion runs can be safely interrupted and resumed
- `main()` — orchestrates the full ingestion loop across all 24 profession directories

**Ingestion flow:**

```
For each profession directory:
  For each PDF resume:
    1. Skip if already in progress file
    2. Extract text with pdfplumber
    3. Skip if empty text
    4. Derive point ID from filename (integer) or generate UUID
    5. Get embedding from Gemini (with retry)
    6. Upsert to Qdrant
    7. Mark as done in progress file
```

---

### 7.5 `main.py` — CLI Entry Point

The non-UI entry point. Sets `JOB_DESCRIPTION` and `PROFESSION` as module-level constants, builds tasks, assembles the `Crew`, and runs it with `Process.sequential`. Saves the final report to `hiring_report.md`.

Used for local development and testing without the Streamlit overhead.

---

### 7.6 `app.py` — Streamlit UI

The primary demo interface. Single-file Streamlit application (~1,100 lines) that wraps the entire pipeline in a polished dark-mode dashboard.

See [Section 11](#11-the-streamlit-ui--feature-guide) for the full UI feature guide.

**Key architectural decisions in `app.py`:**

- The crew runs in a **background thread** (`threading.Thread`) to avoid blocking the Streamlit main thread
- **Two queues** (`log_q`, `result_q`) bridge the background thread to the UI layer — the thread pushes structured messages, the main thread reads them on each re-render cycle
- **Auto-refresh** via `time.sleep(1.2) + st.rerun()` polls the queues every ~1.2 seconds while the crew is running
- `load_dotenv()` is called at the top of `app.py` to ensure all API keys from `.env` are available in Streamlit's process before any crewai imports occur

---

## 8. The Agent Pipeline — Deep Dive

### Context chaining

CrewAI's `context` parameter on a `Task` causes the task's description to be automatically prepended with the output of every listed prior task before the agent sees it. This means:

- The evaluator (Task 3) sees: its own description + the structured requirements (Task 1 output) + all 10 clean profiles (Task 2 output)
- The report writer (Task 4) sees: its own description + the requirements + all 10 scorecards

This eliminates the need for custom prompt engineering to pass context — CrewAI handles it natively.

### Why sequential over hierarchical?

`Process.sequential` runs agents in order. `Process.hierarchical` uses a manager agent to delegate. For this use case, sequential is strictly correct: each step has a single, well-defined input (prior step's output) and a single well-defined output. A manager agent adds latency and unpredictability without any benefit.

### Scoring rubric design

Task 3 instructs the evaluator to weight criteria according to the priority labels assigned by Task 1. This means the weighting is **dynamic** — it adapts to each job description automatically. A job that labels Python as `HIGH` will result in higher Python weight in the scorecard than a job that labels it `MEDIUM`.

---

## 9. RAG & Vector Search — How Retrieval Works

### What is RAG?

RAG (Retrieval-Augmented Generation) is a pattern where relevant documents are retrieved from a database and injected into an LLM's context window, rather than relying solely on the model's trained knowledge. In this system, the "documents" are resumes and the "query" is the job description.

### Vector embeddings

An embedding model converts text into a high-dimensional float vector. Semantically similar texts produce vectors that are geometrically close. `gemini-embedding-001` produces 3,072-dimensional vectors trained for semantic similarity tasks.

### Why cosine similarity?

Cosine distance measures the angle between two vectors, not their length. This means a long, verbose resume and a concise resume covering the same skills will score similarly against a matching query. Euclidean distance would penalize the shorter resume unfairly.

### The collection design

```
Collection: hr_resumes
Vector config: {
  "text_vectors": VectorParams(size=3072, distance=COSINE)
}
Payload fields per point:
  - file_path   (string)
  - profession  (string, e.g. "ENGINEERING")
  - resume_content (string, full raw text)
```

### Scale characteristics

With 2,500 resumes across 24 categories (~104 per category), Qdrant's approximate nearest neighbor (ANN) index returns results in milliseconds. The system scales cleanly to 100,000+ resumes because Qdrant uses HNSW indexing — search time grows logarithmically, not linearly.

### One vector per resume vs. chunking

Chunking splits long documents into overlapping segments and stores each chunk as a separate vector. This is optimal when the unit of retrieval is a passage (e.g., a clause in a 100-page contract). For resumes, the unit of retrieval is a whole person. A resume is 600–900 tokens — well within the embedding model's limit. One vector per resume preserves cross-section relationships (e.g., "senior Python engineer at AWS with ML background") that averaged chunk vectors lose.

---

## 10. LLM Strategy — Resilient Multi-Provider Fallback

### Why four providers?

Production AI systems built on a single API have a single point of failure. Rate limits, outages, and quota exhaustion are common during live demos and testing. The four-provider chain eliminates this risk entirely.

### Provider characteristics

| Provider | Model | Context Window | Speed | Notes |
|---|---|---|---|---|
| Google Gemini | gemini-3.1-pro-preview | 1M tokens | Fast | Primary — best for long-context tasks |
| Cohere | command-a-03-2025 | 256k tokens | Fast | Strong structured output |
| OpenRouter (Hunter) | hunter-alpha | Varies | Varies | Aggregator — routes to available models |
| Groq (LLaMA) | llama-3.3-70b-versatile | 128k tokens | Very fast | Open-source fallback, free tier |

### Fallback cascade logic

```
Crew starts
    │
    ├─► Call Gemini
    │     Success? ──► Use Gemini, continue
    │     Fail (429 / error)? ──► Wait 3s, try next
    │
    ├─► Call Cohere
    │     Success? ──► Switch to Cohere for remaining calls
    │     Fail? ──► Wait 3s, try next
    │
    ├─► Call Hunter (OpenRouter)
    │     Success? ──► Switch to Hunter
    │     Fail? ──► Wait 3s, try next
    │
    └─► Call Groq LLaMA
          Success? ──► Switch to Groq
          Fail? ──► Raise RuntimeError (all exhausted)
```

Once a provider switch occurs, all subsequent calls in that crew run use the new provider. The switch is logged to console so developers can see it happening.

---

## 11. The Streamlit UI — Feature Guide

### Layout

The UI is split into two panels:

**Left Panel** (fixed width, ~420px):
- Job description textarea (pre-filled with the default Senior Data Engineer JD)
- Talent category dropdown (all 24 profession directories)
- Run Screening button (disables during execution to prevent double-runs)
- Four agent status cards (Idle → Active with pulse glow → Done with green accent)
- Gradient progress bar tracking pipeline completion (0% → 25% → 50% → 75% → 100%)
- Live elapsed timer

**Right Panel** (flexible, fills remaining width):
Three tabs:

**📡 Live Feed tab**
- Metrics row: Agents Done / Candidates Retrieved / Elapsed
- Dark terminal-style log window streaming every line of CrewAI's verbose output, color-coded:
  - Cyan: agent actions
  - Indigo: task transitions
  - Green: final outputs
  - Amber: thinking/reasoning lines
  - Red: errors

**👥 Candidates tab**
- Appears as soon as Qdrant retrieval completes (before agents start)
- 10 candidate cards in a 2-column grid
- Each card shows: candidate number, similarity score, preview of first lines
- "View CV" button on each card expands the full raw resume text in a scrollable textarea below the grid

**📋 Report tab**
- Appears when all four agents complete
- Full markdown report rendered with custom CSS (styled tables, headers, blockquotes)
- Download button exports the report as a timestamped `.md` file

### Threading model

```
Streamlit main thread          Background thread
        │                              │
        │  st.button clicked           │
        ├─────────────────────────────►│
        │  Thread starts               │  run_crew_thread()
        │                              │    ├─► retrieve_candidates()
        │  log_q.put("__CANDIDATES__") │◄───┤
        │                              │    ├─► build_tasks()
        │  log_q.put("__AGENT__:0")    │◄───┤
        │                              │    ├─► crew.kickoff()
        │  log_q.put(verbose lines)    │◄───┤   (4 agents, sequential)
        │                              │    │
        │  log_q.put("__DONE__:...")   │◄───┤
        │                              │    └─► result_q.put(report)
        │
   [every 1.2s: st.rerun()]
   [drain queues, update UI state]
```

### Message protocol

The background thread communicates with the UI via structured prefix messages:

| Prefix | Meaning |
|---|---|
| `__PHASE__:text` | Phase status update (shown in log) |
| `__CANDIDATES__:json` | Raw candidate string from Qdrant |
| `__AGENT__:N` | Agent N is now starting (triggers status card update) |
| `__DONE__:report` | Crew complete, report follows |
| `__ERROR__:message` | Pipeline error |
| (no prefix) | Raw CrewAI verbose output line |

---

## 12. Prerequisites & Installation

### System requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| Docker | Any recent | For running Qdrant locally |
| Node.js | Not required | Only needed for docx tools |

### Step 1 — Clone and create environment

```bash
git clone <your-repo>
cd hr-resume-screener
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Step 2 — Install Python dependencies

```bash
pip install crewai qdrant-client pdfplumber google-genai python-dotenv streamlit
```

### Step 3 — Start Qdrant with Docker

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

Verify at: `http://localhost:6333/dashboard`

### Step 4 — Set up resume dataset

```
data/
└── data/
    ├── ENGINEERING/
    │   ├── 10001.pdf
    │   └── 10002.pdf
    ├── FINANCE/
    └── ...
```

### Step 5 — Configure `.env`

```bash
cp .env.example .env
# then fill in your keys (see Section 13)
```

### Step 6 — Run ingestion (once)

```bash
python ingest.py
```

### Step 7 — Launch the app

```bash
streamlit run app.py
```

---

## 13. Configuration — Environment Variables

Create a `.env` file in the project root with the following keys:

```env
# Required — Primary LLM
GEMINI_API_KEY=your_gemini_api_key_here

# Required — Fallback LLMs (get free tier keys from each provider)
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

**Where to get each key:**

| Key | URL | Free Tier |
|---|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey | Yes |
| `GROQ_API_KEY` | https://console.groq.com | Yes |
| `COHERE_API_KEY` | https://dashboard.cohere.com | Yes |
| `OPENROUTER_API_KEY` | https://openrouter.ai/keys | Yes (limited) |

> **Important:** Never commit `.env` to version control. Add it to `.gitignore`.

---

## 14. Running the System

### Option A — Streamlit UI (recommended for demos)

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Option B — Command line

Edit `JOB_DESCRIPTION` and `PROFESSION` in `main.py`, then:

```bash
python main.py
# Output saved to hiring_report.md
```

### Option C — Run ingestion only

```bash
python ingest.py
```

Safe to interrupt and re-run — progress is checkpointed automatically.

---

## 15. Output — The Hiring Report

The final report contains 10 sections in this order:

1. **ROLE REQUIREMENTS RECAP** — 3–5 bullets summarizing non-negotiables, so the committee has a shared reference
2. **EXECUTIVE SUMMARY** — 3–4 sentences: pool quality, top finding, and the single top recommendation (always leads)
3. **CANDIDATE SNAPSHOTS** — one paragraph per candidate: role, experience level, one strength, one gap
4. **RANKED SHORTLIST TABLE** — all 10 candidates: Rank | Name/ID | Score | Verdict | One-Line Note
5. **TOP 3 CANDIDATES — DETAILED PROFILES** — full scorecards, interview focus areas, recommended panel
6. **PIPELINE CANDIDATES** — mid-tier candidates worth keeping warm, with conditions for advancement
7. **NOT RECOMMENDED** — one sentence per candidate explaining the disqualifying factor
8. **RISK & DATA QUALITY NOTES** — scoring uncertainty flags (incomplete profiles, gaps, compensation mismatches)
9. **NEXT STEPS** — concrete action checklist with owners and decision deadline
10. **FINAL RECOMMENDATION** — one decisive paragraph naming the top candidate and the reason

---

## 16. Known Issues & Fixes

### Qdrant connection refused

```
httpx.ConnectError: [Errno 111] Connection refused
```

Qdrant container is not running.

```bash
docker start qdrant
# or first-time setup:
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### Empty results from Qdrant

```
No candidates found in the database for this profession and job description.
```

Either the collection hasn't been ingested yet, or the `PROFESSION` value doesn't match the stored payload values. Check stored values:

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

Expected behavior on the free tier (1,000 req/day limit). The retry logic handles it. If the daily quota is exhausted, stop and re-run the next day — `ingestion_progress.json` will resume from where you left off.

### CrewAI routing to OpenAI with your Gemini key (401 error)

**Symptom:** `Error code: 401 - Incorrect API key provided: AIzaSy...`

**Cause:** Streamlit runs in a fresh Python process that doesn't inherit your terminal's environment variables. If `load_dotenv()` is not called before any `crewai` imports, the `.env` file is never loaded and the Gemini key is `None`. CrewAI's internal LiteLLM layer then attempts an OpenAI call using whatever key it finds, which fails with a 401.

**Fix:** `app.py` calls `load_dotenv()` at the very top of the file, before any imports that trigger CrewAI initialization. If you ever restructure the file, ensure `load_dotenv()` remains at the top.

### Context window exceeded (Groq)

```
Error: context length exceeded
```

Groq's LLaMA 3.3 70B has a 128k context window. Ten full resumes plus task prompts can exceed this. Reduce in `tasks.py`:

```python
TOP_K = 5  # reduce from 10
```

---

## 17. Design Decisions & Rationale

### Why CrewAI over a single LLM call?

A single LLM call with all instructions produces inconsistent results when the prompt is complex. Breaking the work into four focused agents — each with a narrow, well-defined responsibility — produces more reliable, auditable output. Each agent's output can be inspected independently, and failures are isolated.

### Why Qdrant over in-memory comparison?

With 2,500 resumes, passing all of them to an LLM is prohibitively expensive in tokens and latency. Qdrant retrieves only the semantically relevant subset in milliseconds. This is the RAG pattern — retrieval before generation — and it scales to any dataset size.

### Why one vector per resume instead of chunking?

Chunking is appropriate when the retrieval unit is a passage (e.g., a clause in a legal contract). Here the retrieval unit is a whole person. A resume is short enough (600–900 tokens) to embed as a single document, and a whole-document vector preserves cross-section relationships that chunk-average vectors lose.

### Why metadata filtering instead of separate collections per profession?

24 separate collections would require managing 24 connection objects, separate upsert logic, and complex queries if cross-profession search is ever needed. Payload metadata filtering achieves the same pre-filtering effect with a single collection and one line of code.

### Why cosine distance instead of dot product or Euclidean?

Cosine similarity captures semantic direction regardless of vector magnitude. A verbose resume and a concise one covering the same skills produce vectors of different magnitudes but similar directions. Cosine treats them as equally similar to a matching query.

### Why the same embedding model for ingestion and query?

Embedding models produce vectors in their own mathematical space. Mixing models (e.g., embed resumes with Gemini, query with OpenAI) would produce vectors in incompatible spaces — similarity scores would be meaningless. Both ingestion and query use `gemini-embedding-001` with `task_type="SEMANTIC_SIMILARITY"`.

### Why background threading in Streamlit?

Streamlit's execution model re-runs the entire script on every user interaction. A synchronous crew run (which takes several minutes) would freeze the entire UI and make it appear crashed. Running the crew in a `threading.Thread` with queue-based message passing allows the UI to remain responsive and show live progress throughout.

---

## 18. Responsible AI Considerations

### Bias awareness

This system scores candidates against a rubric derived from a job description. The quality and fairness of the output is directly tied to the quality and fairness of the input. A biased job description (e.g., requiring credentials that correlate with demographic groups rather than job performance) will produce biased rankings.

**Mitigation:** The Job Analyst agent is instructed to flag ambiguous or potentially exclusionary language. Users should review the requirements analysis output before acting on the candidate rankings.

### Data quality limits

The evaluator is explicitly instructed to flag scoring uncertainty when candidate profiles have missing or inconsistent data. A high score from an incomplete profile should be treated as a preliminary signal, not a confirmed assessment.

### Human review is mandatory

This system produces a **recommendation**, not a decision. The final hiring decision must be made by humans who have met the candidate, conducted structured interviews, and verified claims. The system is a first-pass screening tool, not a replacement for human judgment.

### Candidate privacy

Resume data is stored locally in Qdrant running on localhost. No candidate data is transmitted to external services except for the embedding API calls to Gemini (text content only, no PII transmitted as structured data). In a production deployment, data handling would need to comply with applicable privacy regulations (GDPR, CCPA, etc.).

---

## 19. Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — a pattern where relevant documents are retrieved from a database and injected into an LLM's context before generation |
| **Vector Embedding** | A numerical representation of text as a high-dimensional float array, where semantically similar texts produce geometrically close vectors |
| **Cosine Similarity** | A distance metric that measures the angle between two vectors, capturing semantic direction regardless of vector magnitude |
| **Qdrant** | An open-source vector database optimized for approximate nearest neighbor search with metadata filtering |
| **HNSW** | Hierarchical Navigable Small World — the graph-based indexing algorithm Qdrant uses for fast approximate nearest neighbor search |
| **CrewAI** | A Python framework for orchestrating multi-agent LLM workflows with defined roles, goals, backstories, and task chains |
| **LiteLLM** | A unified API layer (used internally by CrewAI) that normalizes calls across 100+ LLM providers |
| **Sequential Process** | CrewAI execution mode where agents run in strict order, each receiving all prior outputs as context |
| **ResilientLLM** | The custom wrapper class in `llm.py` that transparently cascades through backup providers on failure |
| **Fallback Chain** | The ordered list of LLM providers tried in sequence: Gemini → Cohere → Hunter → Groq |
| **Profession Filter** | The career category label (e.g., `ENGINEERING`) stored in Qdrant payload, used to narrow the vector search space |
| **TOP_K** | The number of candidate profiles retrieved from Qdrant per query (default: 10) |
| **Scorecard** | The structured evaluation output per candidate: weighted fit score, sub-scores, strengths, gaps, and a verdict |
| **Verdict** | The evaluator's categorical judgment: `STRONG FIT / GOOD FIT / PARTIAL FIT / WEAK FIT` |

---

*Built for NationAI · Applied AI Instructor Demo · March 2026*
