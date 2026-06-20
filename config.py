import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Architectural Thresholds
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.45
MAX_UNRESOLVED_TURNS = 3

# Model Definitions
GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-004"