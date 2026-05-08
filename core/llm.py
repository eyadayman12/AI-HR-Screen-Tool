from crewai import LLM
import time
from core.settings_config import settings

hunter_llm = LLM(
    model = settings.openrouter_model,
    base_url = settings.openrouter_base_url,
    temperature=settings.model_temperature,
)

cohere_llm = LLM(
    model= settings.cohere_model,
    temperature=settings.model_temperature,
)

gemini_llm = LLM(
    model=settings.gemini_model,
    temperature=settings.model_temperature,
)

groq_llm = LLM(
    model = settings.groq_model,
    temperature= settings.model_temperature,
)


FALLBACK_CHAIN = [gemini_llm, cohere_llm, hunter_llm, groq_llm]
FALLBACK_NAMES = ["Gemini 3.1 pro", "Cohere Command-A", "Hunter Alpha (OpenRouter)", "LLaMA 3.3 70B (Groq)"]


class ResilientLLM:

    def __init__(self, starting_index: int = 0):
        self._index = starting_index
        self._active = FALLBACK_CHAIN[self._index]
        self._active_name = FALLBACK_NAMES[self._index]

  
    def __getattr__(self, name):
        return getattr(self._active, name)

    def call(self, messages, **kwargs):
        """
        Attempt the call on the active LLM. On failure, cascade through
        the fallback chain. Raises only if all providers are exhausted.
        """
        for i in range(self._index, len(FALLBACK_CHAIN)):
            llm = FALLBACK_CHAIN[i]
            name = FALLBACK_NAMES[i]
            try:
                result = llm.call(messages, **kwargs)
                if i != self._index:
                    print(f"\n  [ResilientLLM] Switched to: {name} (will use for remaining calls)")
                    self._index = i
                    self._active = llm
                    self._active_name = name
                return result

            except Exception as e:
                error_str = str(e)
                is_rate_limit = any(k in error_str.lower() for k in ("429", "quota", "rate limit", "too many"))
                is_provider_error = any(k in error_str.lower() for k in ("400", "provider returned", "bad request", "stealth"))

                if is_rate_limit:
                    print(f"\n  [ResilientLLM] {name} rate limited. Trying next provider...")
                elif is_provider_error:
                    print(f"\n  [ResilientLLM] {name} provider error: {error_str[:120]}. Trying next...")
                else:
                    print(f"\n  [ResilientLLM] {name} failed: {error_str[:120]}. Trying next...")

                if i == len(FALLBACK_CHAIN) - 1:
                    raise RuntimeError(
                        f"All LLM providers exhausted. Last error from {name}: {e}"
                    )
                time.sleep(3)

        raise RuntimeError("All LLM providers exhausted.")


    @property
    def model(self):
        return self._active.model

    def __repr__(self):
        return f"ResilientLLM(active={self._active_name})"



def get_llm_with_fallback() -> ResilientLLM:
    """
    Probe each LLM in FALLBACK_CHAIN with a minimal test call.
    Return a ResilientLLM starting at the first provider that responds.

    If all probes fail, still return a ResilientLLM starting at index 0
    (Hunter) — the actual failure will be caught and cascaded at call time.
    """
    probe_message = [{"role": "user", "content": "reply with the single word: ok"}]
    for i, (llm, name) in enumerate(zip(FALLBACK_CHAIN, FALLBACK_NAMES)):
        print(f"  Probing {name}...")
        try:
            result = llm.call(probe_message)
            if result:
                print(f"  ✓ {name} is available. Using as primary LLM.\n")
                return ResilientLLM(starting_index=i)
        except Exception as e:
            print(f"  ✗ {name} unavailable: {str(e)[:80]}")
            time.sleep(2)

    print("\n  [WARNING] All LLM probes failed. Starting with Hunter — will cascade on first call.\n")
    return ResilientLLM(starting_index=0)