from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.rag_pipeline import RAGPipeline


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


# =========================================================
# Request Models
# =========================================================

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Customer support question",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of support examples to retrieve",
    )


# =========================================================
# Response Models
# =========================================================

class RetrievedSource(BaseModel):
    customer_query: str
    agent_response: str
    similarity: float


class ChatResponse(BaseModel):
    query: str

    answer: Optional[str]

    decision: str
    escalate: bool

    reason: str

    risk_level: str
    risk_categories: List[str]

    confidence: float
    confidence_level: str

    max_similarity: Optional[float]
    average_similarity: Optional[float]

    retrieved_results: List[RetrievedSource]


# =========================================================
# Pipeline
# =========================================================

pipeline = None


def get_pipeline():
    global pipeline

    if pipeline is None:
        pipeline = RAGPipeline()

    return pipeline


# =========================================================
# Chat Endpoint
# =========================================================

@router.post(
    "",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    try:

        rag = get_pipeline()

        result = rag.generate_response(
            request.query,
            top_k=request.top_k,
        )

        # -------------------------------------------------
        # Clean retrieved results
        # -------------------------------------------------

        sources = []

        for item in result.get(
            "retrieved_results",
            [],
        ):

            sources.append(
                {
                    "customer_query": str(
                        item.get(
                            "customer_query",
                            "",
                        )
                    ),
                    "agent_response": str(
                        item.get(
                            "agent_response",
                            "",
                        )
                    ),
                    "similarity": float(
                        item.get(
                            "similarity",
                            0.0,
                        )
                    ),
                }
            )

        # -------------------------------------------------
        # Build API response
        # -------------------------------------------------

        return ChatResponse(
            query=result["query"],
            answer=result.get("answer"),

            decision=result["decision"],
            escalate=result["escalate"],

            reason=result["reason"],

            risk_level=result["risk_level"],
            risk_categories=result[
                "risk_categories"
            ],

            confidence=float(
                result["confidence"]
            ),
            confidence_level=result[
                "confidence_level"
            ],

            max_similarity=result.get(
                "max_similarity"
            ),
            average_similarity=result.get(
                "average_similarity"
            ),

            retrieved_results=sources,
        )

    except ValueError as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    except Exception as e:

        print(
            f"Chat endpoint error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred "
                "while processing the request."
            ),
        )