import os
from dotenv import load_dotenv
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
K_NEIGHBORS = 3
