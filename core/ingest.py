from qdrant_client import QdrantClient, models
import pdfplumber
from google import genai
from google.genai import types
from core.settings_config import settings

import os
import time
import json

qdrant_config = settings.qdrant_host + ":" + str(settings.qdrant_port)
client = QdrantClient(qdrant_config, timeout=600)
gemini_client = genai.Client()

def sep(n=50):
    print("-" * n)


def collection_creation():
    existing = [c.name for c in client.get_collections().collections]
    if settings.collection_name in existing:
        print(f"Collection '{settings.collection_name}' already exists, skipping creation.")
        return

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config={
            "text_vectors": models.VectorParams(
                size=3072,
                distance=models.Distance.COSINE
            )
        }
    )

    print(f"Collection '{settings.collection_name}' ready.")



def load_progress() -> set:
    """Load the set of already-ingested file paths from the progress file."""
    if os.path.exists(settings.progress_file):
        with open(settings.progress_file, "r") as f:
            return set(json.load(f))
    return set()


def save_progress(ingested: set):
    """Save the set of ingested file paths to disk."""
    with open(settings.progress_file, "w") as f:
        json.dump(list(ingested), f, indent=2)


def get_embedding_with_retry(text: str) -> list:
    """
    Call Gemini embedding API with retry logic.
    - Waits DELAY_BETWEEN_REQUESTS seconds after every call.
    - On rate limit (429) or any API error, waits RETRY_WAIT seconds then retries.
    - Raises after MAX_RETRIES failed attempts.
    """
    for attempt in range(1, settings.max_retries + 1):
        try:
            result = gemini_client.models.embed_content(
                model=settings.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            embeddings = result.embeddings[0].values
            time.sleep(settings.delay_between_requests)
            return embeddings

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "quota" in error_str or "rate" in error_str

            if is_rate_limit:
                print(f"\t  [Rate limit] Attempt {attempt}/{settings.max_retries} — waiting {settings.retry_wait}s before retry...")
                time.sleep(settings.retry_wait)
            else:
                print(f"\t  [API error] Attempt {attempt}/{settings.max_retries} — {e}")
                time.sleep(settings.retry_wait)

            if attempt == settings.max_retries:
                raise RuntimeError(f"Failed after {settings.max_retries} attempts. Last error: {e}")


def qdrant_upsert(id: int, text: str, file_path: str, profession: str) -> None:
    embeddings = get_embedding_with_retry(text)

    client.upsert(
        collection_name=settings.collection_name,
        points=[
            models.PointStruct(
                id=id,
                vector={"text_vectors": embeddings},
                payload={
                    "file_path": file_path,
                    "profession": profession,
                    "resume_content": text
                }
            )
        ]
    )


def extract_resume_text(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


class ResumeIngestion:

    def __init__(self, root_path="data"):
        self.root_path = root_path
        self.resume_dirs = os.listdir(self.root_path)
        self.ingested = load_progress()
        self.failed = []
        self.done_count = len(self.ingested)

    def cal_total_resumes(self):
        return sum(
            len(os.listdir(os.path.join(self.root_path, p)))
            for p in self.resume_dirs
            if os.path.isdir(os.path.join(self.root_path, p))
        )
    
    def load_resume_pdf(self, resume_path, resume, profession, total_resumes):
        if resume_path in self.ingested:
            print(f"\t- [SKIP] {resume} already ingested.")
            return

        if not resume.lower().endswith(".pdf"):
            print(f"\t- [SKIP] {resume} is not a PDF.")
            return

        try:
            resume_text = extract_resume_text(resume_path)
            if not resume_text.strip():
                print(f"\t- [WARN] {resume} extracted empty text, skipping.")
                self.failed.append({"file": resume_path, "reason": "empty text"})
                return
            
            try:
                point_id = int(resume.split(".")[0])
            except ValueError:
                import uuid
                point_id = uuid.uuid4().hex

            qdrant_upsert(
                id=point_id,
                text=resume_text,
                file_path=resume_path,
                profession=profession
            )

            self.ingested.add(resume_path)
            self.done_count += 1
            save_progress(self.ingested)
            print(f"\t- [{self.done_count}/{total_resumes}] {resume} ingested successfully.")

        except Exception as e:
            print(f"\t- [FAILED] {resume} — {e}")
            self.failed.append({"file": resume_path, "reason": str(e)})
            return

    def load_resumes(self, total_resumes):

        for profession in self.resume_dirs:
            
            profession_path = os.path.join(self.root_path, profession)
            if not os.path.isdir(profession_path):
                continue

            resumes = os.listdir(profession_path)
            print(f"Ingesting '{profession}' profession ({len(resumes)} resumes):")

            for resume in resumes:
                resume_path = os.path.join(profession_path, resume)
                self.load_resume_pdf(resume_path, resume, profession, total_resumes)

            sep()

    def ingestion_report(self):

        print(f"\nDone! {self.done_count} ingested, {len(self.failed)} failed.")
        if self.failed:
            print("\nFailed files:")
            for f in self.failed:
                print(f"  - {f['file']}: {f['reason']}")
            with open("failed_ingestions.json", "w") as out:
                json.dump(self.failed, out, indent=2)
            print("\nFailed list saved to failed_ingestions.json")

    def ingest_resumes(self):
        collection_creation()
        sep()
        total_resumes = self.cal_total_resumes()
        print(f"Progress file found: {self.done_count} resumes already ingested, skipping them.")
        sep()
        self.load_resumes(total_resumes)
        self.ingestion_report()

