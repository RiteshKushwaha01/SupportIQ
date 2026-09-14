from pathlib import Path
import pandas as pd


INPUT_PATH = Path("evaluation/response_quality_results.csv")
OUTPUT_PATH = Path("evaluation/response_quality_results.csv")


df = pd.read_csv(INPUT_PATH)


def determine_generation_mode(row):
    answer = row.get("answer")

    # Error rows came from the real Gemini evaluation run.
    if row.get("status") == "error":
        return "gemini"

    # Human escalation does not call the LLM.
    if row.get("decision") == "human":
        return "not_generated"

    # Existing mock responses have this known prefix.
    if isinstance(answer, str) and answer.startswith(
        "This is a development-mode response."
    ):
        return "mock"

    # Existing successful non-mock AI response.
    return "gemini"


df["generation_mode"] = df.apply(determine_generation_mode, axis=1)

df.to_csv(OUTPUT_PATH, index=False)

print("Generation modes added successfully.\n")

print(
    df[
        [
            "golden_index",
            "status",
            "decision",
            "generation_mode",
        ]
    ].to_string(index=False)
)

print("\nGeneration mode counts:")
print(df["generation_mode"].value_counts(dropna=False))