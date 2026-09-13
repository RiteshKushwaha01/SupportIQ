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
from src.escalation import evaluate_escalation


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
        # 3. Initialize Gemini
        # -------------------------------------------------

        self.client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url=(
                "https://generativelanguage.googleapis.com/"
                "v1beta/openai/"
            ),
        )

        print("RAG pipeline ready!")

    # =====================================================
    # RETRIEVAL
    # =====================================================

    def retrieve(self, query, top_k=TOP_K):
        """
        Retrieve the most relevant historical
        customer-support conversations.
        """

        return self.retriever.search(
            query,
            top_k=top_k,
        )

    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    def build_context(self, query, top_k=TOP_K):
        """
        Retrieve relevant examples and construct
        the RAG prompt.
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

    # =====================================================
    # GENERATE RESPONSE
    # =====================================================

    def _generate_with_gemini(self, prompt):
        """
        Generate a response using Gemini.
        """

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
    # COMPLETE RAG PIPELINE
    # =====================================================

    def generate_response(self, query, top_k=TOP_K):
        """
        Complete SupportIQ decision pipeline:

        Customer query
              ↓
        FAISS retrieval
              ↓
        Confidence analysis
              ↓
        Risk analysis
              ↓
        Decision
           /       \
        Human       AI
                    ↓
                  Gemini
        """

        # -------------------------------------------------
        # 1. Retrieve historical examples
        # -------------------------------------------------

        results = self.retrieve(
            query,
            top_k=top_k,
        )

        # -------------------------------------------------
        # 2. Evaluate confidence + risk
        # -------------------------------------------------

        decision = evaluate_escalation(
            query,
            results,
        )

        # -------------------------------------------------
        # 3. Human escalation
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
                "confidence_level": decision["confidence"][
                    "confidence_level"
                ],

                "max_similarity": decision["confidence"].get(
                    "max_similarity"
                ),

                "average_similarity": decision["confidence"].get(
                    "average_similarity"
                ),

                "retrieved_results": results,
            }

        # -------------------------------------------------
        # 4. Build RAG prompt
        # -------------------------------------------------

        prompt = build_prompt(
            query,
            results,
        )

        # -------------------------------------------------
        # 5. Generate AI response
        # -------------------------------------------------

        answer = self._generate_with_gemini(
            prompt
        )

        # -------------------------------------------------
        # 6. Return complete result
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
            "confidence_level": decision["confidence"][
                "confidence_level"
            ],

            "max_similarity": decision["confidence"].get(
                "max_similarity"
            ),

            "average_similarity": decision["confidence"].get(
                "average_similarity"
            ),

            "retrieved_results": results,
        }