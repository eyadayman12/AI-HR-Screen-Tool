from qdrant_client import QdrantClient, models
import pdfplumber
from google import genai
from google.genai import types

import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

get_gemini_api_key = os.getenv("GEMINI_API_KEY")
client = QdrantClient("http://localhost:6333", timeout=600)
model_name = "gemini-embedding-001"
gemini_client = genai.Client(api_key=get_gemini_api_key)
collection_name = "hr_resumes"

DELAY_BETWEEN_REQUESTS = 5
MAX_RETRIES = 5
RETRY_WAIT = 60
PROGRESS_FILE = "ingestion_progress.json"


def sep(n=50):
    print("-" * n)


def collection_creation():
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"Collection '{collection_name}' already exists, skipping creation.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text_vectors": models.VectorParams(
                size=3072,
                distance=models.Distance.COSINE
            )
        }
    )

    print(f"Collection '{collection_name}' ready.")



def load_progress() -> set:
    """Load the set of already-ingested file paths from the progress file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_progress(ingested: set):
    """Save the set of ingested file paths to disk."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(ingested), f, indent=2)


def get_embedding_with_retry(text: str) -> list:
    """
    Call Gemini embedding API with retry logic.
    - Waits DELAY_BETWEEN_REQUESTS seconds after every call.
    - On rate limit (429) or any API error, waits RETRY_WAIT seconds then retries.
    - Raises after MAX_RETRIES failed attempts.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = gemini_client.models.embed_content(
                model=model_name,
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
            )
            embeddings = result.embeddings[0].values
            time.sleep(DELAY_BETWEEN_REQUESTS)
            return embeddings

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "quota" in error_str or "rate" in error_str

            if is_rate_limit:
                print(f"\t  [Rate limit] Attempt {attempt}/{MAX_RETRIES} — waiting {RETRY_WAIT}s before retry...")
                time.sleep(RETRY_WAIT)
            else:
                print(f"\t  [API error] Attempt {attempt}/{MAX_RETRIES} — {e}")
                time.sleep(RETRY_WAIT)

            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed after {MAX_RETRIES} attempts. Last error: {e}")


def qdrant_upsert(id: int, text: str, file_path: str, profession: str) -> None:
    embeddings = get_embedding_with_retry(text)

    client.upsert(
        collection_name=collection_name,
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


def main():
    root_path = os.path.join("data", "data")
    resume_dirs = os.listdir(root_path)

    ingested = load_progress()
    failed = []

    total_resumes = sum(
        len(os.listdir(os.path.join(root_path, p)))
        for p in resume_dirs
        if os.path.isdir(os.path.join(root_path, p))
    )
    done_count = len(ingested)

    print(f"Progress file found: {done_count} resumes already ingested, skipping them.")
    sep()

    for profession in resume_dirs:
        profession_path = os.path.join(root_path, profession)

        if not os.path.isdir(profession_path):
            continue

        resumes = os.listdir(profession_path)
        print(f"Ingesting '{profession}' profession ({len(resumes)} resumes):")

        for resume in resumes:
            resume_path = os.path.join(profession_path, resume)

            if resume_path in ingested:
                print(f"\t- [SKIP] {resume} already ingested.")
                continue

            if not resume.lower().endswith(".pdf"):
                print(f"\t- [SKIP] {resume} is not a PDF.")
                continue

            try:
                resume_text = extract_resume_text(resume_path)

                if not resume_text.strip():
                    print(f"\t- [WARN] {resume} extracted empty text, skipping.")
                    failed.append({"file": resume_path, "reason": "empty text"})
                    continue
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

                ingested.add(resume_path)
                done_count += 1
                save_progress(ingested)
                print(f"\t- [{done_count}/{total_resumes}] {resume} ingested successfully.")

            except Exception as e:
                print(f"\t- [FAILED] {resume} — {e}")
                failed.append({"file": resume_path, "reason": str(e)})
                continue

        sep()

    print(f"\nDone! {done_count} ingested, {len(failed)} failed.")
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {f['file']}: {f['reason']}")
        with open("failed_ingestions.json", "w") as out:
            json.dump(failed, out, indent=2)
        print("\nFailed list saved to failed_ingestions.json")

if __name__ == "__main__":
    collection_creation()
    sep()
    main()