from dotenv import load_dotenv
import os

load_dotenv()

MODEL_NAME = "gemma2:2b"

MAX_RETRIES = 1

TOP_K = 3

MIN_RELEVANCE_SCORE = 0.0

TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)

