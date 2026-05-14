from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import get_logger
import config

logger = get_logger(__name__)


class DataIngestor:
    def __init__(self):
        self.directory_path = config.DATA_PATH
        self.loader_cls_map = {
            ".pdf": PyMuPDFLoader,
            ".txt": TextLoader
        }
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

        logger.info(f"DataIngestor initialized | Path: {self.directory_path}")

    def _load_ext(self, ext_info):
        ext, loader_cls = ext_info
        valid_docs = []

        logger.info(f"Loading files with extension: {ext}")

        try:
            loader = DirectoryLoader(
                self.directory_path,
                glob=f"**/*{ext}",
                loader_cls=loader_cls
            )
            loaded = loader.load()

            valid_docs = [
                doc for doc in loaded if doc.page_content and doc.page_content.strip()
            ]

            logger.info(
                f"{ext} → Loaded: {len(loaded)} | Valid: {len(valid_docs)}")

        except Exception as e:
            logger.error(f"Failed loading {ext} files: {e}", exc_info=True)

        return valid_docs

    def load_and_split(self):
        logger.info("Starting parallel document loading...")

        all_docs = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(
                self._load_ext, self.loader_cls_map.items()))

        for doc_list in results:
            all_docs.extend(doc_list)

        logger.info(f"Total documents before split: {len(all_docs)}")

        chunks = self.splitter.split_documents(all_docs)

        logger.info(f"Total chunks created: {len(chunks)}")

        return chunks
