import pandas as pd
import numpy as np

from src.config import (
    CORPUS_PATH,
    EMBEDDINGS_PATH,
    INDEX_PATH,
    EMBEDDING_MODEL,
    EVALUATION_DIR,
)

from src.retrieval import Retriever


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GOLDEN_PATH = "notebooks/amazonhelp_golden_set_raw.csv"
CONVERSATIONS_PATH = "data/amazonhelp_conversations.csv"

TOP_K = 5

# Confidence thresholds to test
THRESHOLDS = np.arange(0.70, 0.96, 0.01)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def normalize_text(text):
    """
    Normalize text so that small formatting differences
    do not affect exact comparison.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def response_matches(expected_text, retrieved_text):
    """
    Determine whether the retrieved response matches the
    expected response.

    We use normalized exact matching first.
    """

    expected = normalize_text(expected_text)
    retrieved = normalize_text(retrieved_text)

    if not expected or not retrieved:
        return False

    return expected == retrieved


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading golden set...")

golden = pd.read_csv(GOLDEN_PATH)

print(f"Golden set rows: {len(golden)}")
print(f"Golden set columns: {golden.columns.tolist()}")


print("\nLoading conversation data...")

conversations = pd.read_csv(
    CONVERSATIONS_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
    ],
)

print(f"Conversation rows: {len(conversations)}")


# ---------------------------------------------------------
# Build tweet_id -> response text lookup
# ---------------------------------------------------------

print("\nBuilding response lookup...")

response_lookup = (
    conversations
    .drop_duplicates("tweet_id")
    .set_index("tweet_id")["text"]
    .to_dict()
)

print(f"Response lookup size: {len(response_lookup)}")


# ---------------------------------------------------------
# Initialize retriever
# ---------------------------------------------------------

print("\nInitializing retriever...")

retriever = Retriever(
    corpus_path=CORPUS_PATH,
    embeddings_path=EMBEDDINGS_PATH,
    index_path=INDEX_PATH,
    model_name=EMBEDDING_MODEL,
)

print("Retriever ready!")


# ---------------------------------------------------------
# Calibration dataset
# ---------------------------------------------------------

calibration_rows = []

evaluable = 0
non_evaluable = 0


print("\nRunning confidence calibration...")
print("-" * 60)


for i, row in golden.iterrows():

    query = str(row["text"])

    response_id = row["response_tweet_id"]

    # -----------------------------------------------------
    # Some response_tweet_id values can contain multiple IDs
    # -----------------------------------------------------

    if pd.isna(response_id):
        non_evaluable += 1
        continue

    response_id_text = str(response_id).strip()

    # Take the first response ID if multiple IDs exist
    first_response_id = response_id_text.split(",")[0].strip()

    try:
        response_id_int = int(float(first_response_id))
    except ValueError:
        non_evaluable += 1
        continue

    # -----------------------------------------------------
    # Find expected response text
    # -----------------------------------------------------

    expected_response = response_lookup.get(response_id_int)

    if not expected_response:
        non_evaluable += 1
        continue

    # -----------------------------------------------------
    # Retrieve similar support conversations
    # -----------------------------------------------------

    results = retriever.search(
        query,
        top_k=TOP_K,
    )

    if not results:
        non_evaluable += 1
        continue

    # -----------------------------------------------------
    # Calculate confidence
    # -----------------------------------------------------

    similarities = [
        float(result["similarity"])
        for result in results
    ]

    max_similarity = max(similarities)

    average_similarity = sum(similarities) / len(similarities)

    confidence = (
        0.7 * max_similarity
        + 0.3 * average_similarity
    )

    # -----------------------------------------------------
    # Check whether expected response was retrieved
    # -----------------------------------------------------

    retrieved_correct = False

    for result in results:

        retrieved_response = result["agent_response"]

        if response_matches(
            expected_response,
            retrieved_response,
        ):
            retrieved_correct = True
            break

    calibration_rows.append(
        {
            "query": query,
            "expected_response": expected_response,
            "max_similarity": max_similarity,
            "average_similarity": average_similarity,
            "confidence": confidence,
            "retrieved_correct": retrieved_correct,
        }
    )

    evaluable += 1

    if (i + 1) % 25 == 0:
        print(
            f"Processed {i + 1}/{len(golden)} "
            f"| Evaluable: {evaluable}"
        )


# ---------------------------------------------------------
# Save calibration dataset
# ---------------------------------------------------------

calibration_df = pd.DataFrame(calibration_rows)

calibration_path = (
    EVALUATION_DIR /
    "confidence_calibration.csv"
)

calibration_df.to_csv(
    calibration_path,
    index=False,
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("CALIBRATION DATA")
print("=" * 60)

print(f"Total golden queries : {len(golden)}")
print(f"Evaluable queries    : {evaluable}")
print(f"Non-evaluable        : {non_evaluable}")

if evaluable > 0:

    accuracy = calibration_df["retrieved_correct"].mean()

    print(
        f"Retrieval correctness: "
        f"{accuracy:.4f}"
    )

    print(
        f"Confidence mean      : "
        f"{calibration_df['confidence'].mean():.4f}"
    )

    print(
        f"Confidence min       : "
        f"{calibration_df['confidence'].min():.4f}"
    )

    print(
        f"Confidence max       : "
        f"{calibration_df['confidence'].max():.4f}"
    )


# ---------------------------------------------------------
# Evaluate thresholds
# ---------------------------------------------------------

threshold_rows = []


for threshold in THRESHOLDS:

    # AI handles queries above threshold
    # Human handles queries below threshold
    predicted_ai = (
        calibration_df["confidence"] >= threshold
    )

    actual_correct = (
        calibration_df["retrieved_correct"]
    )

    # -----------------------------------------------------
    # AI prediction:
    #
    # confidence >= threshold
    #
    # Actual:
    #
    # retrieved_correct = True
    # -----------------------------------------------------

    tp = (
        predicted_ai & actual_correct
    ).sum()

    fp = (
        predicted_ai & ~actual_correct
    ).sum()

    fn = (
        ~predicted_ai & actual_correct
    ).sum()

    tn = (
        ~predicted_ai & ~actual_correct
    ).sum()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    accuracy = (
        (tp + tn) / evaluable
        if evaluable > 0
        else 0
    )

    ai_rate = (
        predicted_ai.mean()
        if evaluable > 0
        else 0
    )

    human_rate = 1 - ai_rate

    threshold_rows.append(
        {
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "ai_rate": ai_rate,
            "human_rate": human_rate,
        }
    )


threshold_df = pd.DataFrame(threshold_rows)

threshold_path = (
    EVALUATION_DIR /
    "confidence_thresholds.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False,
)


# ---------------------------------------------------------
# Print threshold results
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("CONFIDENCE THRESHOLD EVALUATION")
print("=" * 60)

print(
    threshold_df[
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "ai_rate",
            "human_rate",
        ]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# Find recommended thresholds
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("RECOMMENDED THRESHOLDS")
print("=" * 60)


# Thresholds where AI precision >= 90%
trusted = threshold_df[
    threshold_df["precision"] >= 0.90
]

if len(trusted) > 0:

    best = trusted.sort_values(
        ["f1", "ai_rate"],
        ascending=[False, False],
    ).iloc[0]

    print(
        f"\nBest threshold with precision >= 90%: "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Precision : {best['precision']:.4f}"
    )

    print(
        f"Recall    : {best['recall']:.4f}"
    )

    print(
        f"F1        : {best['f1']:.4f}"
    )

    print(
        f"AI rate   : {best['ai_rate']:.4f}"
    )

    print(
        f"Human rate: {best['human_rate']:.4f}"
    )

else:

    print(
        "No threshold achieved 90% precision."
    )


# ---------------------------------------------------------
# Save complete results
# ---------------------------------------------------------

print("\n")
print("Saved files:")

print(
    f"  {calibration_path}"
)

print(
    f"  {threshold_path}"
)

print("\nCalibration complete.")