import json
import webbrowser
from typing import List, Dict
from langchain_openai import ChatOpenAI
from logger import get_logger
from datetime import datetime
import os
import re

logger = get_logger(__name__)

EVAL_PROMPT = """
You are an expert evaluator of retrieval systems. 
Your task is to compare two retrieval strategies.

QUERY:
{query}

------------------------
STRATEGY A:
{context_a}

------------------------
STRATEGY B:
{context_b}

Evaluate both on:
- Relevance (0-10)
- Coverage (0-10)
- Redundancy (0-10) (penalize duplicates)
- Overall (0-10)

Return ONLY a STRICT JSON object. Do not include markdown code blocks.
{{
  "strategy_a": {{"relevance": int, "coverage": int, "redundancy": int, "overall": int}},
  "strategy_b": {{"relevance": int, "coverage": int, "redundancy": int, "overall": int}},
  "winner": "A" or "B",
  "reason": "short explanation"
}}
"""


class RAGEvaluatorPipeline:
    def __init__(self, model="gpt-4o-mini"):
        # Initializing with temperature 0 for consistent evaluation
        self.llm = ChatOpenAI(model=model, temperature=0)

    def _format_context(self, docs: List[Dict]) -> str:
        """Joins document contents into a single string for the prompt."""
        return "\n\n".join([d.get("content", "") for d in docs])

    def _safe_parse(self, response_text: str) -> Dict:
        """Cleans and parses the LLM response into a dictionary."""
        try:
            # Remove markdown code blocks if the model included them
            clean_text = re.sub(r"```json|```", "", response_text).strip()
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"JSON parse error: {e}")
            return {"error": "Failed to parse LLM response", "raw": response_text}

    def evaluate_single(self, item: Dict) -> Dict:
        """Evaluates a single query against two retrieval strategies."""
        query = item["query"]

        # Extracting contexts from both strategies
        context_a = self._format_context(
            item["strategies"].get("strategy_a_normal", []))
        context_b = self._format_context(
            item["strategies"].get("strategy_b_expanded", []))

        prompt = EVAL_PROMPT.format(
            query=query,
            context_a=context_a,
            context_b=context_b
        )

        logger.info(f"Evaluating Query ID: {item['query_id']}")

        # LLM Invoke returns an AIMessage; we need the .content string
        response = self.llm.invoke(prompt)
        parsed = self._safe_parse(response.content)

        return {
            "query_id": item["query_id"],
            "query": query,
            "evaluation": parsed
        }

    def evaluate_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Runs evaluation over a list of queries."""
        return [self.evaluate_single(item) for item in dataset]

    def summarize(self, results: List[Dict]) -> Dict:
        """Calculates win rates and average scores from results."""
        wins_a, wins_b = 0, 0
        score_a, score_b = 0, 0
        valid = 0

        for r in results:
            e = r["evaluation"]
            if "error" in e:
                continue

            valid += 1
            if e["winner"] == "A":
                wins_a += 1
            else:
                wins_b += 1

            score_a += e["strategy_a"]["overall"]
            score_b += e["strategy_b"]["overall"]

        return {
            "total": len(results),
            "valid": valid,
            "A_wins": wins_a,
            "B_wins": wins_b,
            "avg_A": round(score_a / max(valid, 1), 2),
            "avg_B": round(score_b / max(valid, 1), 2),
        }

    def generate_html_report(self, results: List[Dict], summary: Dict, output_file="report.html"):
        """Generates a visual HTML report and opens it in the browser."""
        rows = ""
        for r in results:
            e = r["evaluation"]
            if "error" in e:
                continue

            rows += f"""
            <tr>
                <td>{r['query_id']}</td>
                <td>{r['query']}</td>
                <td>{e['strategy_a']['overall']}</td>
                <td>{e['strategy_b']['overall']}</td>
                <td style="font-weight: bold; color: {'#2ecc71' if e['winner'] == 'B' else '#3498db'}">{e['winner']}</td>
                <td>{e['reason']}</td>
            </tr>
            """

        html = f"""
        <html>
        <head>
            <title>RAG Evaluation Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; background-color: #f9f9f9; }}
                table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                th, td {{ border: 1px solid #ddd; padding: 12px 15px; text-align: left; }}
                th {{ background-color: #2c3e50; color: white; }}
                tr:hover {{ background-color: #f1f1f1; }}
                .summary-card {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.05); display: inline-block; min-width: 300px; }}
                h1 {{ color: #2c3e50; }}
            </style>
        </head>
        <body>
            <h1>📊 RAG Evaluation Report</h1>
            <div class="summary-card">
                <h2>Performance Summary</h2>
                <p><strong>Total Queries:</strong> {summary['total']}</p>
                <p><strong>Valid Evaluations:</strong> {summary['valid']}</p>
                <p><strong>Strategy A Wins:</strong> {summary['A_wins']}</p>
                <p><strong>Strategy B Wins:</strong> {summary['B_wins']}</p>
                <hr>
                <p><strong>Avg Score A:</strong> {summary['avg_A']} / 10</p>
                <p><strong>Avg Score B:</strong> {summary['avg_B']} / 10</p>
            </div>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Query</th>
                    <th>A Score</th>
                    <th>B Score</th>
                    <th>Winner</th>
                    <th>Reasoning</th>
                </tr>
                {rows}
            </table>
        </body>
        </html>
        """

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)

        # Auto-open the report in the default browser
        webbrowser.open("file://" + os.path.realpath(output_file))
        logger.info(f"Report generated: {output_file}")
