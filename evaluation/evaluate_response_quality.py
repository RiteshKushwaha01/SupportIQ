import json
from pathlib import Path

import pandas as pd

from src.rag_pipeline import RAGPipeline


# =========================================================
# Configuration
# =========================================================

GOLDEN_PATH = Path(
    "notebooks/amazonhelp_golden_set_raw.csv"
)

OUTPUT_PATH = Path(
    "evaluation/response_quality_results.csv"
)

# Number of NEW generations to evaluate.
#
# Keep this small because Gemini free-tier requests
# are limited.
SAMPLE_SIZE = 5

# Continue after the records already generated.
START_INDEX = 20


# =========================================================
# Load golden set
# =========================================================

print("=" * 70)
print("SUPPORTIQ RESPONSE QUALITY EVALUATION")
print("=" * 70)

golden = pd.read_csv(GOLDEN_PATH)

print(f"\nGolden queries: {len(golden)}")

sample = golden.iloc[
    START_INDEX:START_INDEX + SAMPLE_SIZE
].copy()

print(
    f"Generating queries "
    f"{START_INDEX + 1} to "
    f"{START_INDEX + len(sample)}"
)


# =========================================================
# Initialize RAG pipeline
# =========================================================

print("\nInitializing RAG pipeline...")

pipeline = RAGPipeline()

print("Pipeline ready.")


# =========================================================
# Generate responses
# =========================================================

results = []


for local_index, (_, row) in enumerate(
    sample.iterrows(),
    start=START_INDEX + 1,
):

    query = str(row["text"])

    print("\n" + "-" * 70)
    print(f"Query {local_index}")
    print("-" * 70)

    print(f"Customer: {query}")

    try:

        response = pipeline.generate_response(
            query,
            top_k=5,
        )

        answer = response.get("answer")

        decision = response.get("decision")

        confidence = response.get("confidence")

        retrieved_results = response.get(
            "retrieved_results",
            [],
        )

        retrieved_context = []

        for result in retrieved_results:

            retrieved_context.append(
                {
                    "customer_query": result.get(
                        "customer_query"
                    ),
                    "agent_response": result.get(
                        "agent_response"
                    ),
                    "similarity": result.get(
                        "similarity"
                    ),
                }
            )

        results.append(
            {
                "golden_index": local_index,
                "tweet_id": row["tweet_id"],
                "query": query,
                "answer": answer,
                "decision": decision,
                "confidence": confidence,
                "retrieved_context": json.dumps(
                    retrieved_context,
                    ensure_ascii=False,
                ),
                "status": "success",
                "generation_mode": (
                    "not_generated"
                    if answer is None
                    else (
                        "mock"
                        if answer.startswith(
                            "This is a development-mode response."
                        )
                        else "gemini"
                    )
                ),
            }
        )

        print(f"Decision: {decision}")
        print(f"Confidence: {confidence}")

        if answer:
            print(f"Answer: {answer}")
        else:
            print("Answer: Escalated to human")

    except Exception as e:

        error_message = str(e)

        print(f"ERROR: {error_message}")

        results.append(
            {
                "golden_index": local_index,
                "tweet_id": row["tweet_id"],
                "query": query,
                "answer": None,
                "decision": "error",
                "confidence": None,
                "retrieved_context": None,
                "status": "error",
                "generation_mode": "error",
            }
        )


# =========================================================
# Save results
# =========================================================

new_results_df = pd.DataFrame(results)


if OUTPUT_PATH.exists():

    previous_df = pd.read_csv(
        OUTPUT_PATH
    )

    combined_df = pd.concat(
        [
            previous_df,
            new_results_df,
        ],
        ignore_index=True,
    )

    combined_df = combined_df.drop_duplicates(
        subset=["golden_index"],
        keep="last",
    )

else:

    combined_df = new_results_df


combined_df = combined_df.sort_values(
    "golden_index"
)


combined_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
)


# =========================================================
# Summary
# =========================================================

print("\n")
print("=" * 70)
print("GENERATION SUMMARY")
print("=" * 70)

print(
    f"Total saved records: "
    f"{len(combined_df)}"
)

print(
    f"Successful generations: "
    f"{(combined_df['status'] == 'success').sum()}"
)

print(
    f"Errors: "
    f"{(combined_df['status'] == 'error').sum()}"
)

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)