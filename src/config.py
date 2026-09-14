from pathlib import Path
import os
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


# Directories
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"


# Retrieval files
CORPUS_PATH = PROCESSED_DIR / "retrieval_corpus.parquet"
EMBEDDINGS_PATH = PROCESSED_DIR / "retrieval_embeddings.npy"
INDEX_PATH = PROCESSED_DIR / "retrieval.index"


# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# Retrieval
TOP_K = 5


# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_MODEL = "gemini-3.8-flash"

MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"