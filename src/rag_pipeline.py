import time

from openai import OpenAI
from fastembed import TextEmbedding

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
from src.intent_model import load_intent_classifier
from scripts.download_artifacts import ensure_artifacts


class RAGPipeline:

    def __init__(self):
        print("Initializing RAG pipeline...")

        # Ensure large retrieval artifacts are available.
        # This runs only when the RAG pipeline is actually needed,
        # not when FastAPI starts.
        print("Checking deployment artifacts...")
        ensure_artifacts()
        print("Deployment artifacts ready.")

        # Load BGE embedding model once and share it
        # between retrieval and intent classification.
        print("Loading shared embedding model...")
        embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

        self.retriever = Retriever(
            corpus_path=CORPUS_PATH,
            embeddings_path=EMBEDDINGS_PATH,
            index_path=INDEX_PATH,
            model_name=EMBEDDING_MODEL,
            model=embedding_model,
        )

        # Intent classifier
        print("Loading intent classifier...")
        self.intent_classifier = load_intent_classifier(
            model=embedding_model
        )

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

    # ---------------------------------------------------------
    # RETRIEVAL
    # ---------------------------------------------------------

    def retrieve(self, query, top_k=TOP_K):

        return self.retriever.search(
            query,
            top_k=top_k,
        )

    # ---------------------------------------------------------
    # BUILD CONTEXT
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # GEMINI GENERATION
    # ---------------------------------------------------------

    def _generate_with_gemini(self, prompt):

        max_retries = 3

        for attempt in range(max_retries):

            try:

                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional customer "
                                "support assistant for AmazonHelp.\n\n"

                                "Answer customers clearly, politely, "
                                "and concisely.\n\n"

                                "Use the retrieved historical support "
                                "conversations as guidance.\n\n"

                                "Do not invent policies, refunds, "
                                "prices, delivery dates, account "
                                "information, or other unsupported "
                                "facts.\n\n"

                                "Do not claim that you performed an "
                                "action such as issuing a refund, "
                                "changing an account, or modifying "
                                "an order.\n\n"

                                "If the retrieved information is "
                                "insufficient to answer the customer, "
                                "say that you do not have enough "
                                "information and recommend contacting "
                                "customer support."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )

                return response.choices[0].message.content

            except Exception as e:

                error_text = str(e)

                # Retry temporary Gemini errors.
                if "503" in  error_text:

                    if attempt < max_retries - 1:

                        wait_time = 2 ** attempt

                        print(
                            f"Gemini temporarily unavailable "
                            f"(attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {wait_time}s..."
                        )

                        time.sleep(wait_time)

                        continue

                # All retries exhausted or non-retryable error.
                raise

    # ---------------------------------------------------------
    # MOCK GENERATION
    # ---------------------------------------------------------

    def _generate_mock_response(
        self,
        query,
        results,
    ):

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

    # ---------------------------------------------------------
    # MAIN RAG PIPELINE
    # ---------------------------------------------------------

    def generate_response(
        self,
        query,
        top_k=TOP_K,
    ):

        # 1. Classify customer intent.
        intent_result = self.intent_classifier.predict(query)

        intent = intent_result["intent"]
        intent_score = intent_result["semantic_score"]

        # 2. Retrieve relevant historical conversations.
        results = self.retrieve(
            query,
            top_k=top_k,
        )

        # 3. Evaluate safety and confidence.
        decision = evaluate_escalation(
            query,
            results,
        )

        # 4. Escalate high-risk/low-confidence requests.
        if decision["escalate"]:

            return {
                "query": query,
                "answer": None,
                "intent": intent,
                "intent_score": intent_score,
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

        # 5. Generate answer.
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

        # 6. Return complete structured response.
        return {
            "query": query,
            "answer": answer,
            "intent": intent,
            "intent_score": intent_score,
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