from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(
    Path(__file__).resolve().parent.parent / ".env"
)

# ==========================
# LLM
# ==========================

MODEL_NAME = "gemma2:2b"

# ==========================
# GROQ
# ==========================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GROQ_MODEL = "llama-3.3-70b-versatile"

# ==========================
# NEO4J
# ==========================

NEO4J_URI = os.getenv(
    "NEO4J_URI"
)

NEO4J_USERNAME = os.getenv(
    "NEO4J_USERNAME"
)

NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD"
)



# ==========================
# RETRIEVAL
# ==========================

TOP_K = 5

MIN_RELEVANCE_SCORE = 0.25

MAX_RETRIES = 1