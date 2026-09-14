from pathlib import Path
import pandas as pd


INPUT_PATH = Path("evaluation/response_quality_review.csv")
OUTPUT_PATH = Path("evaluation/response_quality_metrics.csv")


df = pd.read_csv(INPUT_PATH)


# Only evaluate real Gemini responses that have
# actually been manually scored.
scored = df[
    (df["generation_mode"] == "gemini")
    & (df["relevance_score"].notna())
    & (df["groundedness_score"].notna())
    & (df["helpfulness_score"].notna())
].copy()


if scored.empty:
    print("No scored Gemini responses found.")
    raise SystemExit(1)


# Calculate overall score if it is missing.
scored["overall_score"] = (
    scored["relevance_score"]
    + scored["groundedness_score"]
    + scored["helpfulness_score"]
) / 3


metrics = {
    "evaluated_gemini_responses": len(scored),
    "relevance_mean": scored["relevance_score"].mean(),
    "groundedness_mean": scored["groundedness_score"].mean(),
    "helpfulness_mean": scored["helpfulness_score"].mean(),
    "overall_mean": scored["overall_score"].mean(),
    "mock_responses_excluded": int(
        (df["generation_mode"] == "mock").sum()
    ),
}


metrics_df = pd.DataFrame([metrics])

metrics_df.to_csv(OUTPUT_PATH, index=False)


print("\nResponse Quality Evaluation")
print("=" * 35)

print(f"Gemini responses evaluated : {metrics['evaluated_gemini_responses']}")
print(f"Mock responses excluded    : {metrics['mock_responses_excluded']}")

print(
    f"Relevance                  : "
    f"{metrics['relevance_mean']:.2f}/5"
)

print(
    f"Groundedness               : "
    f"{metrics['groundedness_mean']:.2f}/5"
)

print(
    f"Helpfulness                : "
    f"{metrics['helpfulness_mean']:.2f}/5"
)

print(
    f"Overall                    : "
    f"{metrics['overall_mean']:.2f}/5"
)

print(f"\nMetrics saved to: {OUTPUT_PATH}")