from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import config
from logger import get_logger

logger = get_logger(__name__)


class LocalEmbeddings:
    def __init__(self, model):
        self.model = SentenceTransformer(model)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text]).tolist()[0]


class VectorManager:
    def __init__(self):
        self.embeddings_wrapper = LocalEmbeddings(config.EMBEDDING_MODEL_NAME)

    def create_or_load_db(self, chunks=None):
        if chunks:
            # Create new DB from documents
            logger.info(
                f"VectorManager initialized | Path: {config.CHROMA_PATH}")
            return Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings_wrapper,
                collection_name="context-aware-rag",
                persist_directory=config.CHROMA_PATH
            )
        else:
            # Load existing DB
            logger.info(
                f"Loading existing vector database | Path: {config.CHROMA_PATH}")
            return Chroma(
                persist_directory=config.CHROMA_PATH,
                embedding_function=self.embeddings_wrapper,
                collection_name="context-aware-rag"
            )
