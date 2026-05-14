from openai import OpenAI
import config


class RAGRetriever:
    def __init__(self, vector_store):
        self.store = vector_store
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def expand_query(self, original_query):
        """Strategy B: AI-Enhanced Retrieval."""
        system_prompt = (
            "You are a technical search expert. Expand the user's query into a "
            "semantically rich prompt with technical synonyms and related concepts."
        )
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original Query: {original_query}\n\nExpanded Search Prompt:"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()  # type: ignore

    def run_comparison(self, query):
        # Strategy A: Normal
        normal_results = self.store.as_retriever(
            search_kwargs={"k": config.K_NEIGHBORS}
        ).invoke(query)

        # Strategy B: Expanded
        expanded_query_text = self.expand_query(query)
        expanded_results = self.store.as_retriever(
            search_kwargs={"k": config.K_NEIGHBORS}
        ).invoke(expanded_query_text)

        return normal_results, expanded_results
