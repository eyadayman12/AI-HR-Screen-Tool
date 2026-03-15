from crewai import LLM
import os
import time
from dotenv import load_dotenv

load_dotenv()

gemini_llm = LLM(
    model="gemini/gemini-3.1-flash-lite-preview",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,       # low temperature = consistent, structured outputs
    max_tokens=8192,
)


groq_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    max_tokens=8192,
)

cohere_llm = LLM(
    model="cohere/command-a-03-2025",
    temperature=0.0,
    api_key=os.getenv("COHERE_API_KEY")
)

hunter_llm = LLM(
    model="openrouter/openrouter/hunter-alpha",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.0
)

def get_llm_with_fallback(max_retries: int = 3, retry_wait: int = 15) -> LLM:
    """
    Try Gemini up to max_retries times.
    If all attempts fail, fall back to Groq LLaMA 3.3 70B.

    Returns the LLM object to assign to agents.
    Note: this does a lightweight connectivity check by attempting
    a minimal completion — if it fails, we fall back before the
    full crew run starts, not mid-run.
    """
    return hunter_llm
    print("Checking primary LLM (Gemini 3.1 Flash lite)...")

    for attempt in range(1, max_retries + 1):
        try:
            test = gemini_llm.call([{"role": "user", "content": "reply with the single word: ok"}])
            if test:
                print(f"Gemini 3.1 Flash lite is available. Using as primary LLM.")
                return gemini_llm
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "quota" in error_str or "rate" in error_str

            if is_rate_limit:
                print(f"  [Attempt {attempt}/{max_retries}] Gemini rate limited — waiting {retry_wait}s...")
            else:
                print(f"  [Attempt {attempt}/{max_retries}] Gemini error: {e}")

            if attempt < max_retries:
                time.sleep(retry_wait)

    print(f"Gemini failed after {max_retries} attempts. Switching to backup: LLaMA 3.3 70B (Groq)")
    return groq_llm
