import json
import re
from pathlib import Path

import pandas as pd
from openai import OpenAI

from src.config import GEMINI_API_KEY, LLM_MODEL


# =========================================================
# Configuration
# =========================================================

INPUT_PATH = Path(
    "evaluation/response_quality_review.csv"
)

OUTPUT_PATH = Path(
    "evaluation/judge_results.csv"
)

JUDGE_MODEL = LLM_MODEL


# =========================================================
# LLM Judge Rubric
# =========================================================

JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator judging an AI customer-support response.

The support agent is answering customers for AmazonHelp.

Evaluate ONLY the provided customer query, retrieved historical
support context, and generated answer.

Do not judge based on outside knowledge.

Score these three dimensions from 1 to 5:

1. Relevance
How directly does the answer address the customer's message?

1 = unrelated
2 = mostly unrelated
3 = partially relevant
4 = relevant
5 = directly addresses the request

2. Groundedness
How well is the answer supported by the retrieved historical context?

1 = unsupported or contradicts context
2 = mostly unsupported
3 = partially supported
4 = well supported
5 = strongly supported by the provided evidence

Penalize invented policies, refunds, prices, delivery dates,
account information, or claims that the agent performed actions.

3. Helpfulness
How useful and actionable is the response for the customer?

1 = not useful
2 = minimally useful
3 = somewhat useful
4 = useful
5 = highly useful and actionable

If the case is clearly inappropriate for AI handling or should
be escalated to a human, do not penalize the response merely for
not attempting to solve a sensitive issue.

Return ONLY valid JSON in this exact structure:

{
  "relevance": <integer 1-5>,
  "groundedness": <integer 1-5>,
  "helpfulness": <integer 1-5>,
  "overall": <number>,
  "reasoning": "<brief explanation>"
}

The overall score must be the arithmetic mean of relevance,
groundedness, and helpfulness, rounded to two decimal places.
"""


# =========================================================
# Helpers
# =========================================================

def extract_json(text: str) -> dict:
    """
    Extract the first JSON object from an LLM response.
    """

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(
            "Judge did not return valid JSON."
        )

    return json.loads(match.group(0))


def validate_score(value, field_name):
    value = int(value)

    if value < 1 or value > 5:
        raise ValueError(
            f"{field_name} must be between 1 and 5."
        )

    return value


def build_judge_prompt(row):
    retrieved_context = row.get(
        "retrieved_context",
        ""
    )

    return f"""
Customer query:
{row["query"]}

Generated support response:
{row["answer"]}

Retrieved historical support context:
{retrieved_context}

Evaluate the generated response using the rubric.
Return only JSON.
"""


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("SUPPORTIQ LLM-AS-JUDGE EVALUATION")
    print("=" * 70)

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Add it to .env before running the judge."
        )

    df = pd.read_csv(INPUT_PATH)

    # Only evaluate successful real Gemini generations.
    candidates = df[
        (df["generation_mode"] == "gemini")
        & (df["answer"].notna())
    ].copy()

    print(f"\nTotal generation records: {len(df)}")
    print(
        f"Eligible Gemini responses: {len(candidates)}"
    )

    if candidates.empty:
        print(
            "\nNo real Gemini responses are available "
            "for judging."
        )
        return

    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url=(
            "https://generativelanguage.googleapis.com/"
            "v1beta/openai/"
        ),
    )

    results = []

    for _, row in candidates.iterrows():

        golden_index = row["golden_index"]

        print("\n" + "-" * 70)
        print(
            f"Judging golden index: {golden_index}"
        )
        print("-" * 70)

        try:

            prompt = build_judge_prompt(row)

            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": JUDGE_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            raw_content = (
                response.choices[0]
                .message
                .content
            )

            scores = extract_json(raw_content)

            relevance = validate_score(
                scores["relevance"],
                "relevance",
            )

            groundedness = validate_score(
                scores["groundedness"],
                "groundedness",
            )

            helpfulness = validate_score(
                scores["helpfulness"],
                "helpfulness",
            )

            overall = round(
                (
                    relevance
                    + groundedness
                    + helpfulness
                )
                / 3,
                2,
            )

            results.append(
                {
                    "golden_index": golden_index,
                    "tweet_id": row["tweet_id"],
                    "query": row["query"],
                    "answer": row["answer"],
                    "relevance": relevance,
                    "groundedness": groundedness,
                    "helpfulness": helpfulness,
                    "overall": overall,
                    "reasoning": scores.get(
                        "reasoning",
                        "",
                    ),
                    "judge_model": JUDGE_MODEL,
                    "status": "success",
                }
            )

            print(
                f"Relevance: {relevance}/5"
            )
            print(
                f"Groundedness: {groundedness}/5"
            )
            print(
                f"Helpfulness: {helpfulness}/5"
            )
            print(
                f"Overall: {overall}/5"
            )

        except Exception as e:

            print(f"ERROR: {e}")

            results.append(
                {
                    "golden_index": golden_index,
                    "tweet_id": row["tweet_id"],
                    "query": row["query"],
                    "answer": row["answer"],
                    "relevance": None,
                    "groundedness": None,
                    "helpfulness": None,
                    "overall": None,
                    "reasoning": str(e),
                    "judge_model": JUDGE_MODEL,
                    "status": "error",
                }
            )

    new_results = pd.DataFrame(results)

    # Merge with existing judge results if present.
    if OUTPUT_PATH.exists():

        previous = pd.read_csv(
            OUTPUT_PATH
        )

        combined = pd.concat(
            [
                previous,
                new_results,
            ],
            ignore_index=True,
        )

        combined = combined.drop_duplicates(
            subset=["golden_index"],
            keep="last",
        )

    else:

        combined = new_results

    combined = combined.sort_values(
        "golden_index"
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    # =====================================================
    # Summary
    # =====================================================

    successful = combined[
        combined["status"] == "success"
    ].copy()

    print("\n")
    print("=" * 70)
    print("JUDGE SUMMARY")
    print("=" * 70)

    print(
        f"Responses judged: {len(successful)}"
    )

    print(
        f"Judge errors: "
        f"{(combined['status'] == 'error').sum()}"
    )

    if not successful.empty:

        print(
            f"Relevance: "
            f"{successful['relevance'].mean():.2f}/5"
        )

        print(
            f"Groundedness: "
            f"{successful['groundedness'].mean():.2f}/5"
        )

        print(
            f"Helpfulness: "
            f"{successful['helpfulness'].mean():.2f}/5"
        )

        print(
            f"Overall: "
            f"{successful['overall'].mean():.2f}/5"
        )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()