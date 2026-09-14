from pathlib import Path
import pandas as pd


INPUT_PATH = Path("evaluation/response_quality_results.csv")
OUTPUT_PATH = Path("evaluation/response_quality_review.csv")


df = pd.read_csv(INPUT_PATH)

# Only successful AI-generated responses are candidates
# for response-quality evaluation.
review_df = df[
    (df["status"] == "success")
    & (df["decision"] == "ai")
].copy()


# Keep the useful information for human evaluation.
review_df = review_df[
    [
        "golden_index",
        "tweet_id",
        "query",
        "answer",
        "generation_mode",
        "confidence",
        "retrieved_context",
    ]
].copy()


# Columns to be filled during human review.
review_df["relevance_score"] = ""
review_df["groundedness_score"] = ""
review_df["helpfulness_score"] = ""
review_df["overall_score"] = ""
review_df["review_notes"] = ""


review_df.to_csv(OUTPUT_PATH, index=False)

print("Response-quality review file created successfully.")
print(f"Output: {OUTPUT_PATH}")
print(f"Rows to review: {len(review_df)}")

print("\nGeneration modes:")
print(review_df["generation_mode"].value_counts())

print("\nReview candidates:")
print(
    review_df[
        [
            "golden_index",
            "generation_mode",
            "query",
        ]
    ].to_string(index=False)
)