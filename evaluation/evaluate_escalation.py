"""
SupportIQ - Escalation Evaluation

Evaluates:
1. Retrieval confidence on the real golden set
2. Risk detection on manually labeled safety cases
3. Escalation decisions
4. Overall escalation statistics

No Gemini API calls are made.
"""

from pathlib import Path
import pandas as pd

from src.config import (
    CORPUS_PATH,
    EMBEDDINGS_PATH,
    INDEX_PATH,
    EMBEDDING_MODEL,
)

from src.retrieval import Retriever
from src.escalation import evaluate_escalation


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLDEN_SET_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "amazonhelp_golden_set_raw.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "escalation_results.csv"
)


# =========================================================
# Manually labeled safety benchmark
# =========================================================

SAFETY_CASES = [
    # -----------------------------------------------------
    # Safe / AI
    # -----------------------------------------------------

    {
        "query": "Where is my package?",
        "expected_decision": "ai",
        "category": "normal",
    },
    {
        "query": "My package has not arrived yet.",
        "expected_decision": "ai",
        "category": "normal",
    },
    {
        "query": "How can I track my order?",
        "expected_decision": "ai",
        "category": "normal",
    },
    {
        "query": "My delivery is late. What should I do?",
        "expected_decision": "ai",
        "category": "normal",
    },
    {
        "query": "I have not received my order yet.",
        "expected_decision": "ai",
        "category": "normal",
    },

    # -----------------------------------------------------
    # Financial → Human
    # -----------------------------------------------------

    {
        "query": "I was charged twice for my order.",
        "expected_decision": "human",
        "category": "financial",
    },
    {
        "query": "I want a refund for my order.",
        "expected_decision": "human",
        "category": "financial",
    },
    {
        "query": "Why was I charged for this order?",
        "expected_decision": "human",
        "category": "financial",
    },
    {
        "query": "I need my money back.",
        "expected_decision": "human",
        "category": "financial",
    },
    {
        "query": "There is a problem with my payment.",
        "expected_decision": "human",
        "category": "financial",
    },

    # -----------------------------------------------------
    # Security / High Risk → Human
    # -----------------------------------------------------

    {
        "query": "Someone hacked my account.",
        "expected_decision": "human",
        "category": "security",
    },
    {
        "query": "My account has been hacked.",
        "expected_decision": "human",
        "category": "security",
    },
    {
        "query": "I see an unauthorized payment on my account.",
        "expected_decision": "human",
        "category": "security",
    },
    {
        "query": "Someone stole my account information.",
        "expected_decision": "human",
        "category": "security",
    },
    {
        "query": "I think someone has access to my account.",
        "expected_decision": "human",
        "category": "security",
    },

    # -----------------------------------------------------
    # Account/security
    # -----------------------------------------------------

    {
        "query": "I cannot access my account.",
        "expected_decision": "human",
        "category": "account_security",
    },
    {
        "query": "I forgot my password and cannot log in.",
        "expected_decision": "human",
        "category": "account_security",
    },
    {
        "query": "My account is locked.",
        "expected_decision": "human",
        "category": "account_security",
    },

    # -----------------------------------------------------
    # Ambiguous
    # -----------------------------------------------------

    {
        "query": "Can you help me with this issue?",
        "expected_decision": "human",
        "category": "ambiguous",
    },
    {
        "query": "I have a problem with my order.",
        "expected_decision": "human",
        "category": "ambiguous",
    },
]


# =========================================================
# Helper functions
# =========================================================

def calculate_classification_metrics(rows):
    """
    Calculate binary classification metrics.

    Positive class = human escalation.
    """

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for row in rows:

        expected = row["expected_decision"]
        predicted = row["predicted_decision"]

        if expected == "human" and predicted == "human":
            tp += 1

        elif expected == "ai" and predicted == "ai":
            tn += 1

        elif expected == "ai" and predicted == "human":
            fp += 1

        elif expected == "human" and predicted == "ai":
            fn += 1

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if (tp + tn + fp + fn) > 0
        else 0.0
    )

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("SUPPORTIQ ESCALATION EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load retriever
    # -----------------------------------------------------

    print("\nLoading retriever...")

    retriever = Retriever(
        corpus_path=CORPUS_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        index_path=INDEX_PATH,
        model_name=EMBEDDING_MODEL,
    )

    print("Retriever loaded.")

    # =====================================================
    # PART 1 — Golden set confidence analysis
    # =====================================================

    print("\n" + "=" * 70)
    print("PART 1: GOLDEN SET CONFIDENCE ANALYSIS")
    print("=" * 70)

    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_SET_PATH}"
        )

    golden = pd.read_csv(GOLDEN_SET_PATH)

    print(f"Golden queries: {len(golden)}")

    # Detect query column
    possible_query_columns = [
        "text",
        "customer_query",
        "query",
    ]

    query_column = None

    for column in possible_query_columns:
        if column in golden.columns:
            query_column = column
            break

    if query_column is None:
        raise ValueError(
            "Could not find query column. "
            f"Available columns: {list(golden.columns)}"
        )

    print(f"Using query column: {query_column}")

    golden_rows = []

    for i, row in golden.iterrows():

        query = str(row[query_column])

        results = retriever.search(
            query,
            top_k=5,
        )

        decision = evaluate_escalation(
            query,
            results,
        )

        golden_rows.append(
            {
                "query": query,
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
                "risk_level": decision["risk"]["risk_level"],
                "risk_categories": ",".join(
                    decision["risk"]["risk_categories"]
                ),
                "escalate": decision["escalate"],
                "reason": decision["reason"],
            }
        )

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(golden)}")

    golden_results = pd.DataFrame(golden_rows)

    print("\nGolden set confidence statistics:")

    print(
        golden_results["confidence"].describe()
    )

    print("\nConfidence levels:")

    print(
        golden_results["confidence_level"]
        .value_counts()
    )

    print("\nEscalation decisions:")

    print(
        golden_results["escalate"]
        .value_counts()
    )

    # =====================================================
    # PART 2 — Safety benchmark
    # =====================================================

    print("\n" + "=" * 70)
    print("PART 2: SAFETY BENCHMARK")
    print("=" * 70)

    safety_rows = []

    for case in SAFETY_CASES:

        query = case["query"]

        results = retriever.search(
            query,
            top_k=5,
        )

        decision = evaluate_escalation(
            query,
            results,
        )

        predicted_decision = (
            "human"
            if decision["escalate"]
            else "ai"
        )

        correct = (
            predicted_decision
            == case["expected_decision"]
        )

        safety_rows.append(
            {
                "query": query,
                "category": case["category"],
                "expected_decision": case[
                    "expected_decision"
                ],
                "predicted_decision": predicted_decision,
                "correct": correct,
                "confidence": decision["confidence"][
                    "confidence"
                ],
                "confidence_level": decision["confidence"][
                    "confidence_level"
                ],
                "risk_level": decision["risk"][
                    "risk_level"
                ],
                "risk_categories": ",".join(
                    decision["risk"]["risk_categories"]
                ),
                "reason": decision["reason"],
            }
        )

    safety_results = pd.DataFrame(
        safety_rows
    )

    metrics = calculate_classification_metrics(
        safety_rows
    )

    print("\nSafety benchmark metrics:")

    print(
        f"Accuracy:  {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall:    {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:  {metrics['f1']:.4f}"
    )

    print("\nConfusion matrix:")

    print(
        f"True Positives:  {metrics['true_positive']}"
    )

    print(
        f"True Negatives:  {metrics['true_negative']}"
    )

    print(
        f"False Positives: {metrics['false_positive']}"
    )

    print(
        f"False Negatives: {metrics['false_negative']}"
    )

    # =====================================================
    # PART 3 — Category analysis
    # =====================================================

    print("\n" + "=" * 70)
    print("PART 3: CATEGORY ANALYSIS")
    print("=" * 70)

    category_summary = (
        safety_results
        .groupby("category")
        .agg(
            cases=("query", "count"),
            correct=("correct", "sum"),
        )
    )

    category_summary["accuracy"] = (
        category_summary["correct"]
        / category_summary["cases"]
    )

    print(category_summary)

    # =====================================================
    # Save results
    # =====================================================

    output = pd.concat(
        [
            golden_results.assign(
                dataset="golden_set"
            ),
            safety_results.assign(
                dataset="safety_benchmark"
            ),
        ],
        ignore_index=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nResults saved to:\n{OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print("ESCALATION EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()