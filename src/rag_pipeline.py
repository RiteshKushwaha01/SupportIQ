from openai import OpenAI

from src.config import (
    CORPUS_PATH,
    EMBEDDINGS_PATH,
    INDEX_PATH,
    EMBEDDING_MODEL,
    TOP_K,
    GEMINI_API_KEY,
    LLM_MODEL,
)

from src.retrieval import Retriever
from src.prompts import build_prompt


class RAGPipeline:

    def __init__(self):
        print("Initializing RAG pipeline...")

        # -------------------------------------------------
        # 1. Initialize retriever
        # -------------------------------------------------

        self.retriever = Retriever(
            corpus_path=CORPUS_PATH,
            embeddings_path=EMBEDDINGS_PATH,
            index_path=INDEX_PATH,
            model_name=EMBEDDING_MODEL,
        )

        # -------------------------------------------------
        # 2. Check Gemini API key
        # -------------------------------------------------

        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Make sure it is added to the .env file."
            )

        # -------------------------------------------------
        # 3. Initialize Gemini through OpenAI-compatible API
        # -------------------------------------------------

        self.client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url=(
                "https://generativelanguage.googleapis.com/"
                "v1beta/openai/"
            ),
        )

        print("RAG pipeline ready!")

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    def retrieve(self, query, top_k=TOP_K):
        """
        Retrieve the most relevant historical
        customer-support conversations.
        """

        return self.retriever.search(
            query,
            top_k=top_k,
        )

    # -----------------------------------------------------
    # Build RAG context
    # -----------------------------------------------------

    def build_context(self, query, top_k=TOP_K):
        """
        Retrieve relevant examples and build
        the final RAG prompt.
        """

        results = self.retrieve(
            query,
            top_k=top_k,
        )

        prompt = build_prompt(
            query,
            results,
        )

        return prompt, results

    # -----------------------------------------------------
    # Generate response
    # -----------------------------------------------------

    def generate_response(self, query, top_k=TOP_K):
        """
        Complete RAG pipeline:

        Customer query
            ↓
        Retrieval
            ↓
        Historical examples
            ↓
        RAG prompt
            ↓
        Gemini
            ↓
        Support response
        """

        # 1. Retrieve relevant conversations
        results = self.retrieve(
            query,
            top_k=top_k,
        )

        # 2. Build RAG prompt
        prompt = build_prompt(
            query,
            results,
        )

        # 3. Generate response using Gemini
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional customer support "
                        "assistant for AmazonHelp.\n\n"
                        "Your job is to help customers clearly, "
                        "politely, and concisely.\n\n"
                        "Use the retrieved historical support "
                        "conversations as guidance.\n\n"
                        "Do not invent policies, refunds, prices, "
                        "delivery dates, account information, or "
                        "other unsupported facts.\n\n"
                        "If the retrieved information is insufficient "
                        "to answer the customer, say that you do not "
                        "have enough information and recommend "
                        "contacting customer support."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        # 4. Extract generated answer
        answer = response.choices[0].message.content

        # 5. Return complete result
        return {
            "query": query,
            "answer": answer,
            "retrieved_results": results,
        }