import json
import logging
import sys
from ingestor import DataIngestor
from vector_store import VectorManager
from retriever import RAGRetriever
from logger import get_logger
from evaluator import RAGEvaluatorPipeline

logger = logging.getLogger(__name__)


def load_queries_from_file(file_path="queries.txt", limit=None):
    """
    Load queries from a text file.

    Args:
        file_path (str): Path to queries file
        limit (int, optional): Max number of queries to return (None = all)

    Returns:
        List[str]: List of queries
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            queries = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        if not queries:
            logger.warning("No valid queries found in file.")
            return []

        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                unique_queries.append(q)
                seen.add(q)

        if limit:
            unique_queries = unique_queries[:limit]

        logger.info(f"Loaded {len(unique_queries)} queries from {file_path}")
        return unique_queries

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []

    except Exception as e:
        logger.error(f"Error reading queries file: {e}")
        return []


def save_to_json(queries, all_normal, all_expanded):
    output = []
    for i, q in enumerate(queries):
        output.append({
            "query_id": i + 1,
            "query": q,
            "strategies": {
                "strategy_a_normal": [{"content": d.page_content} for d in all_normal[i]],
                "strategy_b_expanded": [{"content": d.page_content} for d in all_expanded[i]]
            }
        })
    try:
        with open("retriever_results.json", "w") as f:
            json.dump(output, f, indent=4)
        logger.info("Successfully saved results to retriever_results.json")
    except Exception as e:
        logger.error(f"Failed to save JSON output: {e}")


def main():
    try:
        logger.info("Initializing Data Ingestion...")
        ingestor = DataIngestor()
        chunks = ingestor.load_and_split()
        logger.info(f"Documents split into {len(chunks)} chunks.")

        logger.info("Initializing Vector Manager...")
        vm = VectorManager()
        db = vm.create_or_load_db(chunks)
        logger.info("Vector database created/loaded successfully.")

        retriever = RAGRetriever(db)

        test_queries = load_queries_from_file("queries.txt")

        if not test_queries:
            logger.error("No queries found. Please check queries.txt")
            return

        all_normal_res, all_expanded_res = [], []

        logger.info(f"Starting benchmark for {len(test_queries)} queries...")
        for q in test_queries:
            logger.info(f"Processing query: {q}")
            norm, exp = retriever.run_comparison(q)
            all_normal_res.append(norm)
            all_expanded_res.append(exp)

        save_to_json(test_queries, all_normal_res, all_expanded_res)
        logger.info("Done! retriever_results.json is ready.")

        pipeline = RAGEvaluatorPipeline()

        with open("retriever_results.json", "r") as f:
            data = json.load(f)

        results = pipeline.evaluate_dataset(data)
        summary = pipeline.summarize(results)
        pipeline.generate_html_report(results, summary)

    except Exception as e:
        logger.critical(
            f"Pipeline failed due to unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
