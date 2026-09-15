from pathlib import Path
import json

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.neighbors import KNeighborsClassifier


INPUT_PATH = Path("evaluation/intent_annotation.csv")
OUTPUT_PATH = Path("evaluation/semantic_knn_metrics.json")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RANDOM_STATE = 42
N_SPLITS = 4
BATCH_SIZE = 32


def main():

    print("=" * 70)
    print("BGE SEMANTIC KNN INTENT CLASSIFICATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    df = pd.read_csv(INPUT_PATH)

    df["text"] = df["text"].fillna("").astype(str)
    df["intent"] = df["intent"].fillna("").astype(str)

    df = df[df["intent"].str.strip() != ""].copy()

    texts = df["text"].reset_index(drop=True)
    labels = df["intent"].reset_index(drop=True)

    unique_labels = sorted(labels.unique())

    print(f"Examples: {len(df)}")
    print(f"Intents: {len(unique_labels)}")
    print()

    # ---------------------------------------------------------
    # EMBEDDINGS
    # ---------------------------------------------------------

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")

    embeddings = model.encode(
        texts.tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(f"Embedding shape: {embeddings.shape}")
    print()

    # ---------------------------------------------------------
    # TEST DIFFERENT K VALUES
    # ---------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    k_values = [1, 3, 5, 7]

    results = {}

    for k in k_values:

        print("-" * 70)
        print(f"K = {k}")
        print("-" * 70)

        all_true = []
        all_pred = []

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(
            skf.split(embeddings, labels),
            start=1,
        ):

            X_train = embeddings[train_idx]
            X_test = embeddings[test_idx]

            y_train = labels.iloc[train_idx]
            y_test = labels.iloc[test_idx]

            classifier = KNeighborsClassifier(
                n_neighbors=k,
                metric="cosine",
                weights="distance",
            )

            classifier.fit(X_train, y_train)

            predictions = classifier.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions,
            )

            macro_f1 = f1_score(
                y_test,
                predictions,
                labels=unique_labels,
                average="macro",
                zero_division=0,
            )

            weighted_f1 = f1_score(
                y_test,
                predictions,
                labels=unique_labels,
                average="weighted",
                zero_division=0,
            )

            fold_results.append({
                "fold": fold,
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
            })

            all_true.extend(y_test.tolist())
            all_pred.extend(predictions.tolist())

            print(
                f"Fold {fold}: "
                f"Accuracy={accuracy:.4f}, "
                f"Macro-F1={macro_f1:.4f}, "
                f"Weighted-F1={weighted_f1:.4f}"
            )

        accuracy = accuracy_score(
            all_true,
            all_pred,
        )

        macro_f1 = f1_score(
            all_true,
            all_pred,
            labels=unique_labels,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            all_true,
            all_pred,
            labels=unique_labels,
            average="weighted",
            zero_division=0,
        )

        report = classification_report(
            all_true,
            all_pred,
            labels=unique_labels,
            target_names=unique_labels,
            output_dict=True,
            zero_division=0,
        )

        results[str(k)] = {
            "k": k,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "folds": fold_results,
            "classification_report": report,
        }

        print()
        print(f"Overall Accuracy:    {accuracy:.4f}")
        print(f"Overall Macro-F1:    {macro_f1:.4f}")
        print(f"Overall Weighted-F1: {weighted_f1:.4f}")
        print()

    # ---------------------------------------------------------
    # BEST K
    # ---------------------------------------------------------

    best_k = max(
        results,
        key=lambda k: results[k]["macro_f1"],
    )

    print("=" * 70)
    print("BEST K")
    print("=" * 70)

    print(f"K:                 {best_k}")
    print(
        f"Accuracy:          "
        f"{results[best_k]['accuracy']:.4f}"
    )
    print(
        f"Macro-F1:          "
        f"{results[best_k]['macro_f1']:.4f}"
    )
    print(
        f"Weighted-F1:       "
        f"{results[best_k]['weighted_f1']:.4f}"
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output = {
        "model": "BGE-small + cosine KNN",
        "embedding_model": MODEL_NAME,
        "dataset_size": len(df),
        "num_intents": len(unique_labels),
        "cross_validation": {
            "method": "4-fold stratified cross-validation",
            "random_state": RANDOM_STATE,
        },
        "results_by_k": results,
        "best_k": int(best_k),
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(
        f"Saved results → {OUTPUT_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()