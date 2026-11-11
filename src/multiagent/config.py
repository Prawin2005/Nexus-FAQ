import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Determine model provider ("openai" or "gemini")
configured_provider = os.getenv("MODEL_PROVIDER", "").strip().lower()

if configured_provider in ("openai", "gemini"):
    MODEL_PROVIDER = configured_provider
elif OPENAI_API_KEY:
    MODEL_PROVIDER = "openai"
elif GEMINI_API_KEY:
    MODEL_PROVIDER = "gemini"
else:
    MODEL_PROVIDER = "none"

OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini").strip()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip()

CHROMA_PERSIST_DIR = str(Path(__file__).resolve().parent.parent.parent / "chroma_db")
SQLITE_DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "checkpoints.db")


def log_config_status():
    """Print safe, non-sensitive configuration status at startup."""
    print(f"[CONFIG] Provider: {MODEL_PROVIDER}")
    print(f"[CONFIG] OpenAI key loaded: {'YES' if bool(OPENAI_API_KEY) else 'NO'}")
    print(f"[CONFIG] Gemini key loaded: {'YES' if bool(GEMINI_API_KEY) else 'NO'}")
    if MODEL_PROVIDER == "openai" and not OPENAI_API_KEY:
        print("[CONFIG] WARNING: Provider is 'openai' but OPENAI_API_KEY is not set!")
    elif MODEL_PROVIDER == "gemini" and not GEMINI_API_KEY:
        print("[CONFIG] WARNING: Provider is 'gemini' but GEMINI_API_KEY is not set!")
    elif MODEL_PROVIDER == "none":
        print("[CONFIG] ERROR: No valid LLM API key configured for OpenAI or Gemini.")


log_config_status()
