from openai import OpenAI

from src.config import (
    CORPUS_PATH,
    EMBEDDINGS_PATH,
    INDEX_PATH,
    EMBEDDING_MODEL,
    TOP_K,
    GEMINI_API_KEY,
    LLM_MODEL,
    MOCK_LLM,
)

from src.retrieval import Retriever
from src.prompts import build_prompt
from src.escalation import evaluate_escalation


class RAGPipeline:

    def __init__(self):

        print("Initializing RAG pipeline...")

        self.retriever = Retriever(
            corpus_path=CORPUS_PATH,
            embeddings_path=EMBEDDINGS_PATH,
            index_path=INDEX_PATH,
            model_name=EMBEDDING_MODEL,
        )

        # Gemini is not required in mock mode
        if not MOCK_LLM:

            if not GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY not found. "
                    "Make sure it is added to the .env file."
                )

            self.client = OpenAI(
                api_key=GEMINI_API_KEY,
                base_url=(
                    "https://generativelanguage.googleapis.com/"
                    "v1beta/openai/"
                ),
            )

        else:

            self.client = None

        print("RAG pipeline ready!")

    # =====================================================
    # Retrieval
    # =====================================================

    def retrieve(self, query, top_k=TOP_K):

        return self.retriever.search(
            query,
            top_k=top_k,
        )

    # =====================================================
    # Context
    # =====================================================

    def build_context(self, query, top_k=TOP_K):

        results = self.retrieve(
            query,
            top_k=top_k,
        )

        prompt = build_prompt(
            query,
            results,
        )

        return prompt, results

    # =====================================================
    # Gemini Generation
    # =====================================================

    def _generate_with_gemini(self, prompt):

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional customer support "
                        "assistant for AmazonHelp.\n\n"

                        "Answer customers clearly, politely, "
                        "and concisely.\n\n"

                        "Use the retrieved historical support "
                        "conversations as guidance.\n\n"

                        "Do not invent policies, refunds, prices, "
                        "delivery dates, account information, or "
                        "other unsupported facts.\n\n"

                        "Do not claim that you performed an action "
                        "such as issuing a refund, changing an "
                        "account, or modifying an order.\n\n"

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

        return response.choices[0].message.content

    # =====================================================
    # Mock Generation
    # =====================================================

    def _generate_mock_response(
        self,
        query,
        results,
    ):
        """
        Development-only response generator.

        This allows the API and frontend to be tested
        without consuming Gemini API quota.
        """

        if results:

            best_result = results[0]

            return (
                "This is a development-mode response. "
                "A similar previous support case was found: "
                f"{best_result['customer_query']}"
            )

        return (
            "This is a development-mode response. "
            "No relevant previous support case was found."
        )

    # =====================================================
    # Main Response Generation
    # =====================================================

    def generate_response(
        self,
        query,
        top_k=TOP_K,
    ):

        # -------------------------------------------------
        # Retrieve
        # -------------------------------------------------

        results = self.retrieve(
            query,
            top_k=top_k,
        )

        # -------------------------------------------------
        # Safety / Escalation
        # -------------------------------------------------

        decision = evaluate_escalation(
            query,
            results,
        )

        # -------------------------------------------------
        # Human escalation
        # -------------------------------------------------

        if decision["escalate"]:

            return {
                "query": query,
                "answer": None,

                "decision": "human",
                "escalate": True,

                "reason": decision["reason"],

                "risk_level": decision["risk"]["risk_level"],
                "risk_categories": decision["risk"]["risk_categories"],

                "confidence": decision["confidence"]["confidence"],
                "confidence_level": decision["confidence"]["confidence_level"],

                "max_similarity": decision["confidence"].get(
                    "max_similarity"
                ),

                "average_similarity": decision["confidence"].get(
                    "average_similarity"
                ),

                "retrieved_results": results,
            }

        # -------------------------------------------------
        # Generate AI response
        # -------------------------------------------------

        if MOCK_LLM:

            answer = self._generate_mock_response(
                query,
                results,
            )

        else:

            prompt = build_prompt(
                query,
                results,
            )

            answer = self._generate_with_gemini(
                prompt
            )

        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

        return {
            "query": query,
            "answer": answer,

            "decision": "ai",
            "escalate": False,

            "reason": decision["reason"],

            "risk_level": decision["risk"]["risk_level"],
            "risk_categories": decision["risk"]["risk_categories"],

            "confidence": decision["confidence"]["confidence"],
            "confidence_level": decision["confidence"]["confidence_level"],

            "max_similarity": decision["confidence"].get(
                "max_similarity"
            ),

            "average_similarity": decision["confidence"].get(
                "average_similarity"
            ),

            "retrieved_results": results,
        }