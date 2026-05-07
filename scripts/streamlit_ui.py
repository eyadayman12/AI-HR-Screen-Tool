import streamlit as st
import sys
import threading
import queue
import time
import re
from datetime import datetime
from dotenv import load_dotenv

from crewai import Crew, Process
from core.agents import job_analyst, candidate_retriever, candidate_evaluator, report_writer
from core.tasks import build_tasks, retrieve_candidates

load_dotenv()

st.set_page_config(
    page_title="AI HR Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Base ── */
html, body, [data-testid="stApp"] {
    background: #080C14;
    font-family: 'DM Sans', sans-serif;
    color: #E8EAF0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif; }

/* ── Noise overlay ── */
body::before {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #0D1525 0%, #0A1020 50%, #0E1628 100%);
    border-bottom: 1px solid rgba(64, 190, 255, 0.15);
    padding: 32px 48px 28px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(64,190,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 15%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #FFFFFF;
    margin: 0 0 4px;
    line-height: 1.1;
}
.hero-title span {
    background: linear-gradient(90deg, #40BEFF, #818CF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: rgba(180,190,210,0.7);
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(64,190,255,0.08);
    border: 1px solid rgba(64,190,255,0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-family: 'DM Sans', sans-serif;
    color: #40BEFF;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 12px;
}
.hero-badge .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #40BEFF;
    animation: pulse 1.8s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

/* ── Main layout ── */
.main-layout {
    display: grid;
    grid-template-columns: 420px 1fr;
    gap: 0;
    min-height: calc(100vh - 120px);
}
.left-panel {
    background: #0A0F1C;
    border-right: 1px solid rgba(255,255,255,0.06);
    padding: 32px 28px;
    overflow-y: auto;
}
.right-panel {
    background: #080C14;
    padding: 32px 36px;
    overflow-y: auto;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #40BEFF;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(64,190,255,0.3), transparent);
}

/* ── Input styling ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #E8EAF0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    line-height: 1.6 !important;
    padding: 14px !important;
    transition: border-color 0.2s ease !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: rgba(64,190,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(64,190,255,0.06) !important;
    outline: none !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #E8EAF0 !important;
}
label { color: rgba(200,210,230,0.8) !important; font-size: 0.82rem !important; font-family: 'DM Sans', sans-serif !important; }

/* ── Run button ── */
.stButton > button {
    background: linear-gradient(135deg, #1A6DFF 0%, #0D4FCC 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(26,109,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 32px rgba(26,109,255,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Agent pipeline display ── */
.pipeline-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.agent-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.agent-card.active {
    background: rgba(64,190,255,0.05);
    border-color: rgba(64,190,255,0.25);
    box-shadow: 0 0 20px rgba(64,190,255,0.08);
}
.agent-card.active::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #40BEFF, #818CF8);
    border-radius: 3px 0 0 3px;
}
.agent-card.done {
    background: rgba(52,211,153,0.04);
    border-color: rgba(52,211,153,0.2);
}
.agent-card.done::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: #34D399;
    border-radius: 3px 0 0 3px;
}
.agent-icon {
    width: 36px; height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    background: rgba(255,255,255,0.05);
}
.agent-card.active .agent-icon { background: rgba(64,190,255,0.12); }
.agent-card.done .agent-icon { background: rgba(52,211,153,0.12); }
.agent-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #C8D0E0;
    margin-bottom: 2px;
}
.agent-card.active .agent-name { color: #FFFFFF; }
.agent-status {
    font-size: 0.73rem;
    color: rgba(150,160,180,0.7);
    font-family: 'DM Sans', sans-serif;
}
.agent-card.active .agent-status { color: #40BEFF; }
.agent-card.done .agent-status { color: #34D399; }

/* ── Live log ── */
.log-container {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 16px;
    height: 220px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 0.72rem;
    line-height: 1.7;
    margin-top: 16px;
}
.log-line { margin: 0; padding: 1px 0; }
.log-line.info { color: #64748B; }
.log-line.agent { color: #40BEFF; }
.log-line.task { color: #818CF8; }
.log-line.output { color: #34D399; }
.log-line.error { color: #F87171; }
.log-line.thinking { color: #FBBF24; }

/* ── Progress bar ── */
.progress-track {
    height: 3px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    margin: 20px 0;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #40BEFF, #818CF8);
    transition: width 0.5s ease;
}

/* ── Candidate cards ── */
.candidates-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 14px;
    margin-top: 8px;
}
.candidate-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 20px;
    cursor: pointer;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.candidate-card:hover {
    background: rgba(64,190,255,0.05);
    border-color: rgba(64,190,255,0.25);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.candidate-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(64,190,255,0.4), transparent);
    opacity: 0;
    transition: opacity 0.25s;
}
.candidate-card:hover::before { opacity: 1; }
.candidate-num {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #40BEFF;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.candidate-score {
    position: absolute;
    top: 16px; right: 16px;
    background: rgba(64,190,255,0.1);
    border: 1px solid rgba(64,190,255,0.2);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #40BEFF;
}
.candidate-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.92rem;
    font-weight: 700;
    color: #E8EAF0;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding-right: 60px;
}
.candidate-title {
    font-size: 0.78rem;
    color: rgba(150,160,180,0.8);
    font-family: 'DM Sans', sans-serif;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.candidate-expand-hint {
    margin-top: 10px;
    font-size: 0.68rem;
    color: rgba(100,116,139,0.7);
    font-family: 'DM Sans', sans-serif;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* ── CV Modal ── */
.cv-modal-overlay {
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.85);
    backdrop-filter: blur(8px);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 32px;
}
.cv-modal {
    background: #0E1525;
    border: 1px solid rgba(64,190,255,0.2);
    border-radius: 20px;
    width: 100%;
    max-width: 760px;
    max-height: 85vh;
    overflow-y: auto;
    padding: 36px 40px;
    position: relative;
    box-shadow: 0 40px 120px rgba(0,0,0,0.8), 0 0 60px rgba(64,190,255,0.06);
}
.cv-modal-close {
    position: absolute;
    top: 20px; right: 20px;
    width: 32px; height: 32px;
    border-radius: 8px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 1rem;
    color: rgba(200,210,230,0.7);
    transition: all 0.2s;
}
.cv-modal-close:hover { background: rgba(248,113,113,0.15); border-color: rgba(248,113,113,0.3); color: #F87171; }
.cv-content {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    line-height: 1.75;
    color: #C8D0E0;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Report section ── */
.report-container {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 36px 40px;
}
.report-container h1, .report-container h2, .report-container h3 {
    font-family: 'Syne', sans-serif;
    color: #FFFFFF;
}
.report-container h1 { font-size: 1.5rem; font-weight: 800; border-bottom: 1px solid rgba(64,190,255,0.2); padding-bottom: 12px; margin-bottom: 24px; }
.report-container h2 { font-size: 1.1rem; font-weight: 700; color: #40BEFF; margin-top: 32px; margin-bottom: 12px; }
.report-container h3 { font-size: 0.95rem; font-weight: 600; color: #E8EAF0; margin-top: 20px; }
.report-container p, .report-container li { font-family: 'DM Sans', sans-serif; font-size: 0.875rem; line-height: 1.75; color: #C8D0E0; }
.report-container table { width: 100%; border-collapse: collapse; margin: 16px 0; }
.report-container th { background: rgba(64,190,255,0.1); color: #40BEFF; font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; padding: 10px 14px; text-align: left; border-bottom: 1px solid rgba(64,190,255,0.2); }
.report-container td { padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.04); font-family: 'DM Sans', sans-serif; font-size: 0.82rem; color: #C8D0E0; }
.report-container tr:hover td { background: rgba(255,255,255,0.02); }
.report-container code { background: rgba(64,190,255,0.08); border: 1px solid rgba(64,190,255,0.15); border-radius: 4px; padding: 2px 6px; font-size: 0.8rem; color: #40BEFF; }
.report-container blockquote { border-left: 3px solid #40BEFF; padding-left: 16px; margin: 16px 0; color: rgba(200,210,230,0.7); font-style: italic; }

/* ── Status states ── */
.status-idle { text-align: center; padding: 60px 20px; color: rgba(100,116,139,0.6); }
.status-idle .idle-icon { font-size: 3rem; margin-bottom: 16px; opacity: 0.4; }
.status-idle p { font-family: 'DM Sans', sans-serif; font-size: 0.85rem; }

/* ── Metrics row ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 24px;
}
.metric-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.68rem;
    color: rgba(150,160,180,0.6);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'DM Sans', sans-serif;
}

/* ── Download btn ── */
.download-wrapper .stDownloadButton > button {
    background: rgba(52,211,153,0.1) !important;
    border: 1px solid rgba(52,211,153,0.3) !important;
    color: #34D399 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    width: auto !important;
    padding: 8px 16px !important;
}

/* ── Spinner ── */
@keyframes spin { to { transform: rotate(360deg); } }
.spin { animation: spin 1s linear infinite; display: inline-block; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(150,160,180,0.6) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    color: #40BEFF !important;
    background: rgba(64,190,255,0.06) !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: #40BEFF !important; }
[data-testid="stMarkdownContainer"] p { font-family: 'DM Sans', sans-serif !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.18); }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Alert / info ── */
[data-testid="stAlert"] {
    background: rgba(64,190,255,0.06) !important;
    border: 1px solid rgba(64,190,255,0.2) !important;
    border-radius: 10px !important;
    color: #C8D0E0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── PROFESSION LIST ───────────────────────────────────────────────────────────
PROFESSIONS = [
    "ALL", "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL",
    "ARTS", "AUTOMOBILE", "AVIATION", "BANKING",
    "BPO", "BUSINESS-DEVELOPMENT", "CHEF", "CONSTRUCTION",
    "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA", "ENGINEERING",
    "FINANCE", "FITNESS", "HEALTHCARE", "HR",
    "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER",
]

AGENTS_META = [
    {"icon": "🔍", "name": "Job Requirements Analyst", "short": "Analyzing job description…"},
    {"icon": "📂", "name": "Talent Database Specialist", "short": "Cleaning candidate profiles…"},
    {"icon": "⚖️",  "name": "Objective Candidate Evaluator", "short": "Scoring all candidates…"},
    {"icon": "📝", "name": "Executive Report Writer", "short": "Writing hiring report…"},
]

DEFAULT_JD = """We are seeking a certified Personal Trainer and Fitness Coach to join our wellness center.

Requirements:
- 2+ years of experience as a personal trainer or fitness coach
- Certified by a recognized body (ACE, NASM, ACSM, or equivalent)
- Experience designing personalized workout and nutrition programs
- Strong knowledge of anatomy, physiology, and exercise science
- Experience working with diverse clients including beginners and athletes

Nice to have:
- Specialization in strength and conditioning or sports performance
- CPR and First Aid certification
- Experience with group fitness classes or online coaching

The role involves conducting fitness assessments, creating tailored training
plans, motivating clients to reach their goals, and tracking progress."""

# ─── SESSION STATE INIT ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "running": False,
        "done": False,
        "log_lines": [],
        "agent_states": [{"status": "idle"} for _ in AGENTS_META],
        "current_agent": -1,
        "progress": 0,
        "candidates": [],
        "report_md": "",
        "selected_candidate": None,
        "elapsed": 0,
        "start_time": None,
        "error": None,
        "jd": DEFAULT_JD,
        "profession": "ALL",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── STDOUT INTERCEPTOR ────────────────────────────────────────────────────────
class StreamCapture:
    def __init__(self, log_queue):
        self._queue = log_queue
        self._original = sys.stdout

    def write(self, text):
        self._original.write(text)
        if text.strip():
            self._queue.put(text)

    def flush(self):
        self._original.flush()


def classify_log(line: str):
    l = line.lower()
    if any(x in l for x in ["agent:", "crew agent", "working on task", "i am ", "as a"]):
        return "agent"
    if any(x in l for x in ["task output:", "final answer:", "result:", "## ", "# "]):
        return "output"
    if any(x in l for x in ["error", "failed", "exception", "traceback"]):
        return "error"
    if any(x in l for x in ["thinking", "analyzing", "evaluating", "writing", "scoring", "retrieving"]):
        return "thinking"
    if any(x in l for x in ["task:", "starting task", "executing task"]):
        return "task"
    return "info"


def detect_agent_from_log(line: str) -> int:
    l = line.lower()
    if any(x in l for x in ["job analyst", "job requirements", "analyzing job", "task 1", "senior job"]):
        return 0
    if any(x in l for x in ["talent database", "candidate retriever", "retrieving", "presenting candidates", "task 2"]):
        return 1
    if any(x in l for x in ["evaluator", "scoring", "scorecard", "evaluation", "task 3", "objective candidate"]):
        return 2
    if any(x in l for x in ["report writer", "hiring report", "writing report", "executive", "task 4"]):
        return 3
    return -1


# ─── CREW RUNNER ──────────────────────────────────────────────────────────────
def run_crew_thread(jd: str, profession: str, log_q: queue.Queue, result_q: queue.Queue):
    try:
        log_q.put("__PHASE__:Querying Qdrant vector database…")
        candidates_raw = retrieve_candidates(jd, profession)
        log_q.put(f"__CANDIDATES__:{candidates_raw}")
        log_q.put("__PHASE__:Candidates retrieved. Launching agent pipeline…")

        tasks = build_tasks(job_description=jd, profession=profession)

        crew = Crew(
            agents=[job_analyst, candidate_retriever, candidate_evaluator, report_writer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        log_q.put("__AGENT__:0")
        result = crew.kickoff()
        log_q.put("__DONE__:" + str(result.raw))
        result_q.put({"status": "ok", "report": str(result.raw)})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log_q.put(f"__ERROR__:{e}")
        result_q.put({"status": "error", "message": str(e), "traceback": tb})


def parse_candidates(raw: str) -> list:
    """Parse the raw candidate text into structured dicts."""
    blocks = re.split(r'CANDIDATE\s+(\d+)', raw)
    candidates = []
    for i in range(1, len(blocks), 2):
        num = blocks[i]
        content = blocks[i + 1] if i + 1 < len(blocks) else ""
        score_match = re.search(r'similarity score:\s*([\d.]+)', content)
        score = score_match.group(1) if score_match else "N/A"
        # Try to extract first non-empty meaningful line as "name"
        lines = [l.strip() for l in content.split('\n') if l.strip() and not l.strip().startswith('---') and 'similarity' not in l.lower() and 'profession:' not in l.lower()]
        first_lines = lines[:3] if lines else ["Candidate " + num]
        candidates.append({
            "num": num,
            "score": score,
            "preview": first_lines,
            "full": content.strip(),
        })
    return candidates

# ─── HERO HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-badge"><span class="dot"></span>AI-Powered · Multi-Agent · RAG</div>
  <div class="hero-title">AI HR <span>Screener</span></div>
  <div class="hero-subtitle">CrewAI · Qdrant Vector Search · LLM Ensemble · Semantic Matching</div>
</div>
""", unsafe_allow_html=True)

# ─── TWO-COLUMN LAYOUT ────────────────────────────────────────────────────────
col_left, col_right = st.columns([4, 7], gap="small")

# ═══ LEFT PANEL ═══════════════════════════════════════════════════════════════
with col_left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)

    # — Input section —
    st.markdown('<div class="section-label">⚙ Configuration</div>', unsafe_allow_html=True)

    jd = st.text_area(
        "Job Description",
        value=st.session_state.jd,
        height=260,
        placeholder="Paste the full job description here…",
        key="jd_input",
    )

    profession = st.selectbox(
        "Talent Category",
        options=PROFESSIONS,
        index=PROFESSIONS.index(st.session_state.profession),
        key="profession_input",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    run_col, _ = st.columns([1, 0.01])
    with run_col:
        run_label = "⏳ Running…" if st.session_state.running else "🚀 Run Screening"
        run_clicked = st.button(run_label, disabled=st.session_state.running)

    if run_clicked and not st.session_state.running:
        # Reset state
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.log_lines = []
        st.session_state.agent_states = [{"status": "idle"} for _ in AGENTS_META]
        st.session_state.current_agent = -1
        st.session_state.progress = 0
        st.session_state.candidates = []
        st.session_state.report_md = ""
        st.session_state.selected_candidate = None
        st.session_state.error = None
        st.session_state.start_time = time.time()
        st.session_state.jd = jd
        st.session_state.profession = profession

        # Stash queues in session state (picklable workaround)
        log_q = queue.Queue()
        result_q = queue.Queue()
        st.session_state["_log_q"] = log_q
        st.session_state["_result_q"] = result_q

        t = threading.Thread(
            target=run_crew_thread,
            args=(jd, profession, log_q, result_q),
            daemon=True,
        )
        t.start()
        st.session_state["_thread"] = t
        st.rerun()

    # — Agent pipeline status —
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🤖 Agent Pipeline</div>', unsafe_allow_html=True)

    for i, meta in enumerate(AGENTS_META):
        state = st.session_state.agent_states[i]
        css_cls = "agent-card"
        status_text = "Waiting…"
        if state["status"] == "active":
            css_cls += " active"
            status_text = meta["short"]
        elif state["status"] == "done":
            css_cls += " done"
            status_text = "✓ Complete"

        st.markdown(f"""
        <div class="{css_cls}">
          <div class="agent-icon">{meta['icon']}</div>
          <div>
            <div class="agent-name">{meta['name']}</div>
            <div class="agent-status">{status_text}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # — Progress bar —
    pct = st.session_state.progress
    st.markdown(f"""
    <div class="progress-track">
      <div class="progress-fill" style="width:{pct}%"></div>
    </div>
    """, unsafe_allow_html=True)

    # — Elapsed —
    if st.session_state.running and st.session_state.start_time:
        elapsed = int(time.time() - st.session_state.start_time)
        st.markdown(f"""
        <div style="text-align:center; font-size:0.72rem; color:rgba(100,116,139,0.7); font-family:'DM Sans',sans-serif;">
          ⏱ Elapsed: {elapsed}s
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ═══ RIGHT PANEL ══════════════════════════════════════════════════════════════
with col_right:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    # ── Process queue messages ──
    if st.session_state.running:
        log_q = st.session_state.get("_log_q")
        result_q = st.session_state.get("_result_q")

        if log_q:
            try:
                while True:
                    msg = log_q.get_nowait()

                    if msg.startswith("__AGENT__:"):
                        idx = int(msg.split(":")[1])
                        # mark previous done
                        if idx > 0:
                            st.session_state.agent_states[idx - 1]["status"] = "done"
                        st.session_state.agent_states[idx]["status"] = "active"
                        st.session_state.current_agent = idx
                        st.session_state.progress = int((idx / 4) * 100)

                    elif msg.startswith("__CANDIDATES__:"):
                        raw = msg[len("__CANDIDATES__:"):]
                        st.session_state.candidates = parse_candidates(raw)

                    elif msg.startswith("__DONE__:"):
                        report = msg[len("__DONE__:"):]
                        st.session_state.report_md = report
                        for i in range(4):
                            st.session_state.agent_states[i]["status"] = "done"
                        st.session_state.progress = 100
                        st.session_state.running = False
                        st.session_state.done = True

                    elif msg.startswith("__ERROR__:"):
                        st.session_state.error = msg[len("__ERROR__:"):]
                        st.session_state.running = False

                    elif msg.startswith("__PHASE__:"):
                        text = msg[len("__PHASE__:"):]
                        st.session_state.log_lines.append(("task", text))

                    else:
                        # Detect agent switches from verbose output
                        agent_idx = detect_agent_from_log(msg)
                        if agent_idx >= 0 and agent_idx != st.session_state.current_agent:
                            if st.session_state.current_agent >= 0:
                                st.session_state.agent_states[st.session_state.current_agent]["status"] = "done"
                            st.session_state.agent_states[agent_idx]["status"] = "active"
                            st.session_state.current_agent = agent_idx
                            st.session_state.progress = int(((agent_idx + 0.5) / 4) * 100)

                        cls = classify_log(msg)
                        st.session_state.log_lines.append((cls, msg.strip()))

            except queue.Empty:
                pass

        # Check result queue
        if result_q:
            try:
                res = result_q.get_nowait()
                if res["status"] == "ok":
                    st.session_state.report_md = res["report"]
                    for i in range(4):
                        st.session_state.agent_states[i]["status"] = "done"
                    st.session_state.progress = 100
                    st.session_state.running = False
                    st.session_state.done = True
                else:
                    st.session_state.error = res["message"]
                    st.session_state.running = False
            except queue.Empty:
                pass

    # ── TABS ──
    tab_labels = ["📡 Live Feed", "👥 Candidates", "📋 Report"]
    tab1, tab2, tab3 = st.tabs(tab_labels)

    # ── TAB 1: LIVE FEED ──
    with tab1:
        if not st.session_state.running and not st.session_state.done and not st.session_state.error:
            st.markdown("""
            <div class="status-idle">
              <div class="idle-icon">🤖</div>
              <p>Configure your job description and talent category,<br>then hit <strong>Run Screening</strong> to launch the agent pipeline.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Metrics row
            elapsed_display = "—"
            if st.session_state.start_time:
                elapsed_display = f"{int(time.time() - st.session_state.start_time)}s"
            if st.session_state.done and st.session_state.start_time:
                elapsed_display = f"{int(time.time() - st.session_state.start_time)}s"

            n_done = sum(1 for a in st.session_state.agent_states if a["status"] == "done")
            n_cands = len(st.session_state.candidates)

            st.markdown(f"""
            <div class="metrics-row">
              <div class="metric-box">
                <div class="metric-value">{n_done}/4</div>
                <div class="metric-label">Agents Done</div>
              </div>
              <div class="metric-box">
                <div class="metric-value">{n_cands}</div>
                <div class="metric-label">Candidates</div>
              </div>
              <div class="metric-box">
                <div class="metric-value">{elapsed_display}</div>
                <div class="metric-label">Elapsed</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.error:
                st.error(f"**Pipeline error:** {st.session_state.error}")
            elif st.session_state.done:
                st.success("✅ All agents completed successfully. Report is ready.")

            # Live log
            log_html_lines = []
            for cls, line in st.session_state.log_lines[-120:]:
                safe = line.replace("<", "&lt;").replace(">", "&gt;")
                log_html_lines.append(f'<p class="log-line {cls}">{safe}</p>')

            log_content = "\n".join(log_html_lines) if log_html_lines else '<p class="log-line info">Waiting for agent output…</p>'

            st.markdown(f"""
            <div class="section-label">📟 Agent Verbose Output</div>
            <div class="log-container" id="log-container">
              {log_content}
            </div>
            <script>
              var lc = document.getElementById('log-container');
              if (lc) lc.scrollTop = lc.scrollHeight;
            </script>
            """, unsafe_allow_html=True)

    # ── TAB 2: CANDIDATES ──
    with tab2:
        if not st.session_state.candidates:
            st.markdown("""
            <div class="status-idle">
              <div class="idle-icon">👥</div>
              <p>Candidate profiles will appear here after<br>the Qdrant vector search completes.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="section-label">👥 Retrieved Candidates — {len(st.session_state.candidates)} profiles from Qdrant</div>', unsafe_allow_html=True)
            st.markdown('<div class="candidates-grid">', unsafe_allow_html=True)

            # We'll use Streamlit columns to fake a grid
            cols = st.columns(2)
            for i, cand in enumerate(st.session_state.candidates):
                col = cols[i % 2]
                with col:
                    preview_name = cand["preview"][0][:40] if cand["preview"] else f"Candidate {cand['num']}"
                    preview_title = cand["preview"][1][:40] if len(cand["preview"]) > 1 else "Resume Profile"

                    card_html = f"""
                    <div class="candidate-card" style="margin-bottom:10px;">
                      <div class="candidate-num">Candidate #{cand['num']}</div>
                      <div class="candidate-score">sim {cand['score']}</div>
                      <div class="candidate-name">{preview_name}</div>
                      <div class="candidate-title">{preview_title}</div>
                      <div class="candidate-expand-hint">▼ Click to expand CV</div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

                    btn_key = f"cv_btn_{i}"
                    if st.button(f"View CV #{cand['num']}", key=btn_key, use_container_width=True):
                        st.session_state.selected_candidate = i

            st.markdown('</div>', unsafe_allow_html=True)

            # — CV Expander Panel —
            if st.session_state.selected_candidate is not None:
                idx = st.session_state.selected_candidate
                if 0 <= idx < len(st.session_state.candidates):
                    cand = st.session_state.candidates[idx]
                    st.markdown("---")
                    st.markdown(f"""
                    <div style="background:rgba(64,190,255,0.04); border:1px solid rgba(64,190,255,0.2); border-radius:16px; padding:28px 32px; margin-top:12px;">
                      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                        <div>
                          <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#FFFFFF;">
                            Candidate #{cand['num']}
                          </div>
                          <div style="font-size:0.75rem; color:rgba(150,160,180,0.7); margin-top:2px;">
                            Similarity Score: <span style="color:#40BEFF; font-weight:600;">{cand['score']}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.text_area(
                        f"CV Content — Candidate #{cand['num']}",
                        value=cand["full"],
                        height=400,
                        key=f"cv_text_{idx}",
                    )

                    if st.button("✕ Close", key="close_cv"):
                        st.session_state.selected_candidate = None
                        st.rerun()

    # ── TAB 3: REPORT ──
    with tab3:
        if not st.session_state.report_md:
            st.markdown("""
            <div class="status-idle">
              <div class="idle-icon">📋</div>
              <p>The hiring report will appear here<br>once all four agents have completed their work.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            header_row = st.columns([5, 1])
            with header_row[0]:
                st.markdown('<div class="section-label">📋 Executive Hiring Report</div>', unsafe_allow_html=True)
            with header_row[1]:
                st.markdown('<div class="download-wrapper">', unsafe_allow_html=True)
                st.download_button(
                    label="⬇ Download",
                    data=st.session_state.report_md,
                    file_name=f"hiring_report_{st.session_state.profession}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    key="dl_btn",
                )
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.markdown(st.session_state.report_md)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── AUTO-REFRESH WHILE RUNNING ───────────────────────────────────────────────
if st.session_state.running:
    time.sleep(1.2)
    st.rerun()
