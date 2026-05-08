[![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CrewAI](https://img.shields.io/badge/CrewAI-multi--agent-7C3AED)](https://www.crewai.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector--search-DC2626?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Gemini](https://img.shields.io/badge/Gemini-embeddings%20%2B%20LLM-4285F4?logo=google&logoColor=white)](https://aistudio.google.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-16A34A)](LICENSE)
[![CI](https://github.com/eyadayman12/AI-HR-Screen-Tool/actions/workflows/ci.yml/badge.svg)](https://github.com/eyadayman12/AI-HR-Screen-Tool/actions)

**A multi-agent AI system that screens and ranks job candidates using RAG, vector search, and a sequential CrewAI pipeline.**

---

## ⚡ Quick Start (Docker)

The fastest way to run the full stack — Qdrant, the API, and all dependencies — in one command. No Python environment setup required.

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed.

**Step 1 — Clone the repo and create your `.env` file:**

```bash
git clone https://github.com/eyadayman12/AI-HR-Screen-Tool.git
cd AI-HR-Screen-Tool
cp .env.example .env          # then open .env and fill in your API keys
```

Your `.env` file needs these two keys at minimum:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

Get them free at [aistudio.google.com](https://aistudio.google.com) (Gemini) and [console.groq.com](https://console.groq.com) (Groq).

**Step 2 — Start all services:**

```bash
docker compose up
```

This starts Qdrant (vector database) and the screening API together. Qdrant's dashboard will be available at `http://localhost:6333/dashboard`. The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

**Step 3 — Run the ingestion pipeline (once, to populate the vector database):**

```bash
# In a separate terminal, while docker compose is running:
docker compose exec api python ingest.py
```

This embeds your resume PDFs and stores them in Qdrant. It runs once and is resumable — safe to stop and restart. With 2,500 resumes on Gemini's free tier, expect ~3.5 hours spread across 3 days (1,000 requests/day limit). Progress is saved automatically.

**Step 4 — Screen your first batch of candidates:**

```bash
curl -X POST http://localhost:8000/screen \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "We are looking for a Senior Data Engineer with 5+ years of experience building ETL pipelines, strong Python skills, and familiarity with cloud platforms (AWS or GCP).",
    "profession": "ENGINEERING",
    "top_k": 10
  }'
```

The endpoint returns a `job_id` immediately. Poll for your results:

```bash
curl http://localhost:8000/jobs/{job_id}
```

When status is `"complete"`, the response includes the full ranked hiring report as structured JSON, plus a `hiring_report.md` saved to the `outputs/` volume.

> **No Docker?** See the [manual installation guide](#6-installation) further below for step-by-step setup with a local Python environment.

---

# Repository Structure

This document explains every file and folder in the AI HR Resume Screener codebase — what each one does, why it exists, and how the pieces connect to each other. It covers both the current state of the repository and the planned structure as the project grows toward a production-ready API with Docker support.

---

## Current structure at a glance

```
AI-HR-Screen-Tool/
│
├── Core pipeline (the 5 modules that run the system)
│   ├── main.py                     ← entry point: configure + launch the crew
│   ├── agents.py                   ← defines the 4 CrewAI agent personas
│   ├── tasks.py                    ← defines the 4 tasks + Qdrant retrieval
│   ├── llm.py                      ← LLM configuration + Gemini/Groq fallback
│   └── ingest.py                   ← Phase 1: PDF extraction + vector DB population
│
├── Data (your resume library — not committed to git)
│   └── data/
│       └── data/
│           ├── ACCOUNTING/         ← one folder per career category
│           ├── ENGINEERING/        ← folder name = "profession" field in Qdrant
│           ├── FINANCE/
│           └── ... (24 total)
│
├── Auto-generated at runtime (not committed to git)
│   ├── hiring_report.md            ← the final output produced by the crew
│   ├── ingestion_progress.json     ← checkpoint file: tracks which PDFs are done
│   └── failed_ingestions.json      ← log of PDFs that failed to embed
│
├── Config / secrets (never committed to git)
│   └── .env                        ← your API keys live here
│
└── Documentation
    └── README.md                   ← main project documentation
```

---

## The 5 core modules in depth

Understanding this codebase is easiest if you think about the **direction of data flow**. These five files form a chain, and data passes through them in a strict order each time you run a screening job. Here is the chain in plain language:

```
.env → llm.py → agents.py → tasks.py → main.py → hiring_report.md
                                 ↑
                            Qdrant (populated by ingest.py, run separately)
```

Reading the files in the order below follows that chain and makes the most sense.

---

### `llm.py` — the base of the chain

This is the first file that executes at runtime, because `agents.py` imports from it at the module level (meaning the import itself triggers the LLM resolution). Its job is to answer one question before anything else runs: which language model will we use for this session?

The file defines four LLM objects using CrewAI's `LLM` class (which wraps LiteLLM under the hood): `gemini_llm` as the primary, `groq_llm` as the open-source fallback, and `cohere_llm` and `hunter_llm` as optional alternatives. The function `get_llm_with_fallback()` probes the primary LLM with a lightweight test call, retries up to three times, and returns the Groq model if Gemini is unavailable.

**One thing to fix in this file:** there is currently an early `return hunter_llm` at the top of `get_llm_with_fallback()` that bypasses the entire probe-and-fallback logic. This was added for testing purposes. Remove that line to restore the intended behaviour.

**What other files depend on this:** `agents.py` imports `get_llm_with_fallback` and calls it once at the top of the module, storing the result in a variable called `llm`. All four agents then receive that same resolved instance. The LLM is chosen once and shared — there is no per-agent or mid-run switching.

---

### `agents.py` — the four reasoning personas

This file defines the four CrewAI `Agent` objects that make up the pipeline. Each agent is a configuration object: it has a `role` (a job title that anchors the persona), a `goal` (a precise description of the output it must produce), and a `backstory` (narrative context that shapes how the LLM approaches the task). None of these contain actual logic — they are prompt engineering constructs that tell the LLM how to behave.

The four agents are:

`job_analyst` acts as a Senior Job Requirements Analyst. It receives the raw job description and its sole responsibility is to produce a structured breakdown of must-have skills, preferred skills, experience levels, and key responsibilities. Its output becomes the shared reference document that every other agent consults.

`candidate_retriever` acts as a Talent Database Specialist. It receives the 10 raw resume texts that were pre-retrieved from Qdrant by `tasks.py`, along with the job analyst's structured requirements. Its job is to clean and structure each candidate's profile — extracting name, skills, experience, and education — without scoring or judging anyone. It is deliberately kept neutral at this stage.

`candidate_evaluator` acts as an Objective Candidate Evaluator. It receives the clean profiles from the retriever and applies a consistent rubric to score each candidate from 0 to 10. The backstory explicitly instructs the LLM to treat every CV as a set of evidence points — no gut feelings, no names, no bias. The output is a scorecard for each candidate with an overall score, a skills match score, an experience score, strengths, gaps, and a one-line verdict.

`report_writer` acts as an Executive Talent Report Writer. It receives all scorecards and produces the final `hiring_report.md` — a polished, scannable document that a hiring committee can read in three minutes. The backstory deliberately models the persona of a former Chief People Officer who knows that busy executives need the recommendation on the first page.

All four agents have `allow_delegation=False`. This is important: it prevents any agent from handing tasks to another agent mid-run, which would break the sequential pipeline guarantee and make the system non-deterministic.

**What other files depend on this:** `main.py` imports all four agent objects by name to assemble the `Crew`.

---

### `tasks.py` — the work orders and the Qdrant bridge

This file does two distinct things and it is worth understanding them separately.

The first thing it does is define the `retrieve_candidates()` function, which is the only place in the codebase that talks to Qdrant at query time. It embeds the job description using the same Gemini embedding model used during ingestion (this is non-negotiable — you must use the same model on both sides, or the vectors live in different mathematical spaces and similarity scores become meaningless). It then calls `qdrant.search()` with a `profession` metadata filter and retrieves the top-K most semantically similar candidates using cosine similarity. The results are formatted into a multi-line string that gets injected directly into Task 2's description.

The second thing it does is define `build_tasks()`, which creates all four `Task` objects. Each task has a `description` (the instructions given to the assigned agent), an `expected_output` (what the agent should produce), an `agent` (which of the four handles it), and a `context` list. The `context` parameter is how tasks pass their outputs forward: Task 3 (evaluator) lists Task 1 and Task 2 in its context, so it automatically receives both the structured requirements and the clean profiles as inputs. Task 4 (report writer) lists all prior tasks. This chaining is what makes the pipeline work without any manual wiring.

**What other files depend on this:** `main.py` calls `build_tasks()` and passes the result to the `Crew`. `tasks.py` in turn imports the four agent objects from `agents.py`.

**One constant to know about:** `TOP_K = 10` at the top of the file controls how many candidates are retrieved from Qdrant. If you encounter context window errors (most likely on Groq's free tier, which has a lower per-request limit than the 128k window suggests), reduce this to 5.

---

### `main.py` — the entry point and the only file you edit per job

This is the file an HR manager would interact with in the current CLI version. It contains two hardcoded variables at the top — `JOB_DESCRIPTION` and `PROFESSION` — which is what you customize for each screening run. Everything else is automated.

The `main()` function assembles the `Crew` object with all four agents and tasks, sets `process=Process.sequential` (tasks run in strict order, each one feeding into the next), and calls `crew.kickoff()`. The result is a string containing the final hiring report, which is both printed to stdout and saved to `hiring_report.md`.

One of the roadmap items is to replace the hardcoded variables here with `argparse` so that the tool can be called from the command line without editing source code: `python main.py --profession ENGINEERING --top-k 5 --job-desc-file jd.txt`. This is a small change with a large quality-of-life impact.

**What other files depend on this:** nothing — `main.py` is the top of the chain. It imports from `agents.py` and `tasks.py` but nothing imports from it.

---

### `ingest.py` — the one-time setup script

This file runs separately from the main pipeline and only needs to be run once (or when you add new resumes to the database). It handles the offline ingestion phase: extracting text from your 2,500 PDF resumes, embedding each one with Gemini Embedding 001, and storing the resulting vectors in Qdrant with metadata.

The embedding model produces 3,072-dimensional float vectors. Each vector is stored in a Qdrant collection named `hr_resumes` under a named vector key `"text_vectors"`. Alongside the vector, three metadata fields are stored as a payload: `profession` (the directory name, which becomes the filter key at query time), `resume_content` (the full extracted text, which is passed to the agents), and `file_path` (the original location on disk).

The file includes robust rate-limit handling for Gemini's free tier (100 requests/minute, 1,000 per day). After every successful embedding it writes the file path to `ingestion_progress.json`, so if the script stops for any reason — a rate limit, a power cut, hitting the daily quota — you can re-run it and it will skip all already-processed files and pick up where it left off.

**What other files depend on this:** nothing at runtime. `ingest.py` is a standalone preparation script. The Qdrant collection it populates is then read by `tasks.py` at query time, but there is no direct code dependency between the two files.

---

## The data folder

```
data/
└── data/
    ├── ACCOUNTING/
    │   ├── 10001.pdf
    │   ├── 10002.pdf
    │   └── ...
    ├── ENGINEERING/
    ├── FINANCE/
    ├── FITNESS/
    └── ... (24 career categories total)
```

The nested `data/data/` path is a quirk of how the dataset was originally downloaded. The outer `data/` folder is the repo-level container; the inner `data/` is from the dataset's own packaging. This is harmless but worth knowing so you do not get confused when reading `ingest.py`'s path traversal code.

The folder names are significant. They become the `profession` field stored in Qdrant during ingestion, and they must exactly match the `PROFESSION` value you set in `main.py` when running a query. The match is case-sensitive. If the folder is named `ENGINEERING`, then `PROFESSION = "engineering"` will return zero results.

**This entire folder should be in `.gitignore`.** PDF files are large, and 2,500 of them would make the repository unusable for anyone who tries to clone it. The data is the user's responsibility to obtain and place locally.

---

## Auto-generated files

These three files are created by the system at runtime. They should all be in `.gitignore` and should never be committed to the repository.

`hiring_report.md` is the final output of each screening run, written by `main.py` after the crew finishes. It is overwritten on every run, so if you want to preserve a report, rename it or move it before running again. One of the roadmap items is to add a timestamped output directory so historical reports are not lost.

`ingestion_progress.json` is written by `ingest.py` after each successful PDF embedding. It contains a list of file paths that have already been processed. The script reads this list at startup and skips any files that appear in it, making ingestion safely resumable. Do not delete this file mid-ingestion unless you want to re-embed everything from scratch.

`failed_ingestions.json` is also written by `ingest.py` and contains file paths where embedding failed after all retry attempts. These are typically PDFs that are corrupted, password-protected, or empty. You can inspect this file after ingestion to decide whether to fix or discard those files.

---

## Config and secrets

`**.env**` holds your API keys and should never be committed to git. The `.gitignore` file should explicitly list `.env` to make this hard to accidentally violate. The file should contain at minimum:

```
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

Optionally you can also add `OPENROUTER_API_KEY` and `COHERE_API_KEY` for the alternative LLMs defined in `llm.py`, but they are not needed for the default pipeline.

**`.env.example`** is a file that should exist in the repo but currently does not. It should be a copy of `.env` with placeholder values instead of real keys, along with comments explaining where to obtain each one. Its purpose is to tell a new contributor exactly what credentials they need without exposing yours. Create this file and commit it.

---

## What is missing from the current structure

Based on the roadmap, the following files and folders will be added as the project matures. This section serves as a map of what is coming so that the structure does not feel incomplete — it is intentionally being built in stages.

**`requirements.txt`** — currently missing. Anyone who clones the repo has no record of which packages to install beyond what is documented in the README. Add this immediately with `pip freeze > requirements.txt` (after activating your virtual environment) or maintain it manually. For a tighter dependency specification, `requirements.txt` for runtime and `requirements-dev.txt` for testing and development tools is a good pattern.

**`LICENSE`** — currently missing. The README badge says MIT but without an actual `LICENSE` file in the root, GitHub does not recognise the license and will not display it in the repository sidebar. Create a `LICENSE` file with the standard MIT text.

**`.gitignore`** — currently missing from what is visible. A proper `.gitignore` for this project should exclude `.env`, `data/`, `*.pdf`, `hiring_report.md`, `ingestion_progress.json`, `failed_ingestions.json`, `__pycache__/`, `*.pyc`, `venv/`, `.env`, and (once added) `outputs/`.

**`api/`** — planned in Phase 2 of the roadmap. This folder will contain the FastAPI application that wraps the CrewAI pipeline behind REST endpoints. The intended structure is `api/main.py` (the FastAPI app and route definitions), `api/schemas.py` (the Pydantic models for request and response validation), and `api/dependencies.py` (shared FastAPI dependencies like API key auth and rate limiting).

**`tests/`** — planned alongside the API work. This folder will contain `pytest` tests for the API endpoints using `httpx`'s async test client. The most important tests to write first are the health check endpoint, input validation (invalid profession name, job description that is too short or too long), and a mocked screening run that does not actually call the LLM or Qdrant.

**`Dockerfile` and `docker-compose.yml`** — planned in Phase 3. The `Dockerfile` will use a multi-stage build (builder stage installs dependencies, runtime stage copies only what is needed to run). The `docker-compose.yml` will define three services: `qdrant` with a named volume for data persistence, `api` (your FastAPI app), and `redis` for background task queuing. These two files live in the repository root.

**`.dockerignore`** — created alongside the Dockerfile. It should exclude `data/`, `*.pdf`, `venv/`, `.env`, `__pycache__`, `*.pyc`, `ingestion_progress.json`, `failed_ingestions.json`, and `hiring_report.md`. Without this file, Docker will send all 2,500 PDFs into the build context on every `docker build`, making builds extremely slow.

**`.github/workflows/`** — planned as the CI/CD layer. At minimum, two workflow files: `ci.yml` (runs `pytest` on every push and pull request) and `docker-publish.yml` (builds and pushes the Docker image to GitHub Container Registry on every push to `main`).

**`docs/`** — planned as the home for static assets referenced in the README. The first thing to add here is a PNG or SVG version of the architecture diagram, replacing the ASCII art currently in the README with an embedded image.

**`outputs/`** — planned as a timestamped output directory for hiring reports. Instead of overwriting `hiring_report.md` on every run, the system will write to `outputs/ENGINEERING_2026-05-08_143022.md` (or `.json` once structured output is added). This folder should be in `.gitignore` but should contain a `.gitkeep` file so the empty directory is tracked by git and created automatically on clone.

---

## Target structure after the roadmap is complete

This is what the repository will look like once all five roadmap phases are finished. Use this as a reference when making structural decisions along the way.

```
AI-HR-Screen-Tool/
│
├── Core pipeline
│   ├── main.py                         ← CLI entry point (argparse-driven)
│   ├── agents.py                       ← 4 CrewAI agent definitions
│   ├── tasks.py                        ← 4 tasks + Qdrant retrieval
│   ├── llm.py                          ← LLM config + fallback resolution
│   └── ingest.py                       ← one-time PDF ingestion script
│
├── API layer (Phase 2)
│   └── api/
│       ├── main.py                     ← FastAPI app: routes, startup, middleware
│       ├── schemas.py                  ← Pydantic models: request + response types
│       └── dependencies.py            ← API key auth, rate limiter, Qdrant client
│
├── Tests (Phase 2)
│   └── tests/
│       ├── conftest.py                 ← shared fixtures (mock crew, mock Qdrant)
│       ├── test_health.py             ← health check endpoint tests
│       ├── test_screen.py             ← screening endpoint: valid + invalid inputs
│       └── test_professions.py        ← professions listing endpoint tests
│
├── Docker (Phase 3)
│   ├── Dockerfile                      ← multi-stage build: builder + runtime
│   ├── docker-compose.yml             ← qdrant + api + redis services
│   └── .dockerignore                   ← excludes data/, PDFs, venv, .env
│
├── CI/CD (Phase 3)
│   └── .github/
│       └── workflows/
│           ├── ci.yml                  ← pytest on push + pull request
│           └── docker-publish.yml      ← build + push to GHCR on main
│
├── Documentation assets
│   └── docs/
│       └── architecture.png           ← visual architecture diagram for README
│
├── Generated outputs (gitignored, but directory tracked)
│   └── outputs/
│       ├── .gitkeep                    ← keeps the empty folder in git
│       └── ENGINEERING_2026-05-08.md  ← example timestamped hiring report
│
├── Data (gitignored — too large to commit)
│   └── data/
│       └── data/
│           ├── ENGINEERING/
│           ├── FINANCE/
│           └── ... (24 career categories)
│
├── Runtime files (gitignored — auto-generated)
│   ├── ingestion_progress.json
│   └── failed_ingestions.json
│
├── Config (gitignored for secrets, committed for templates)
│   ├── .env                            ← your actual API keys (never commit)
│   └── .env.example                   ← placeholder template (commit this)
│
├── Dependency management
│   ├── requirements.txt               ← runtime dependencies
│   └── requirements-dev.txt          ← dev + test dependencies (pytest, httpx)
│
├── Project metadata
│   ├── README.md                       ← main documentation
│   ├── REPO_STRUCTURE.md              ← this document
│   └── LICENSE                         ← MIT license text
│
└── Git config
    └── .gitignore
```

---

## Quick reference: which files to edit vs which to leave alone

As a rule of thumb, the files you will touch regularly are `main.py` (to change the job description and profession for each run), `llm.py` (to switch the active LLM or fix the early return), and `agents.py` (if you want to refine an agent's goal or backstory). The files you will rarely touch are `tasks.py` (unless you are adding a new task or changing `TOP_K`) and `ingest.py` (unless you are adding a new career category to the dataset). The auto-generated files (`hiring_report.md`, `ingestion_progress.json`, `failed_ingestions.json`) should never be edited manually.

When the API layer is added, `api/schemas.py` becomes the most frequently edited file as the output schema evolves, and `api/main.py` is where you add new endpoints. The core pipeline files (`agents.py`, `tasks.py`) remain stable and are only modified when the pipeline's behaviour needs to change — not when new API features are added.

---

*Last updated: May 2026 — reflects the current codebase at commit 8 on `main` and the planned structure through Phase 3 of the project roadmap.*