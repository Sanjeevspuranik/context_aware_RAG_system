import json
import logging
import sys
from ingestor import DataIngestor
from vector_store import VectorManager
from retriever import RAGRetriever
from logger import get_logger
from evaluator import RAGEvaluatorPipeline

logger = logging.getLogger(__name__)


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

        test_queries = [
            "How do government interventions in telecommunications, such as internet shutdowns or app bans, redefine the concept of the public interest?",
            "Explain the impact of non-conservative forces on the total mechanical energy of a system during a physical transformation.",
            "Compare the efficiency and mechanism of ATP production between aerobic respiration and anaerobic fermentation.",
            "How do the principles of fluid viscosity and Poiseuille's Law apply to the human circulatory system during periods of physical exertion?",
            "What are the primary differences in how normative and empirical political science address the legitimacy of institutional power?",
            "What are medusa and polyps in the context of marine biology, and how do their life cycles differ?",
            "Explain adaptive radiation.",
            "explain coeloms and acoelomates.",
            "explain magnetism and how it works.",
            "What is otto cycle and how does it work?"
        ]

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
