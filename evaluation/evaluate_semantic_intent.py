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


INPUT_PATH = Path("evaluation/intent_annotation.csv")
OUTPUT_PATH = Path("evaluation/semantic_intent_metrics.json")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RANDOM_STATE = 42
N_SPLITS = 4
BATCH_SIZE = 32


def create_centroids(embeddings, labels):
    """
    Create one normalized centroid for each intent.
    """
    centroids = {}

    for intent in np.unique(labels):
        class_embeddings = embeddings[labels == intent]

        centroid = class_embeddings.mean(axis=0)

        # Normalize centroid
        norm = np.linalg.norm(centroid)

        if norm > 0:
            centroid = centroid / norm

        centroids[intent] = centroid

    return centroids


def predict_from_centroids(embeddings, centroids):
    """
    Predict the intent with the highest cosine similarity.
    """
    intents = list(centroids.keys())

    centroid_matrix = np.vstack(
        [centroids[intent] for intent in intents]
    )

    # Embeddings are normalized, so dot product = cosine similarity
    similarities = embeddings @ centroid_matrix.T

    predictions = [
        intents[index]
        for index in np.argmax(similarities, axis=1)
    ]

    return predictions


def main():

    print("=" * 70)
    print("SEMANTIC INTENT CLASSIFICATION")
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

    print("Class distribution:")
    print(labels.value_counts())
    print()

    # ---------------------------------------------------------
    # LOAD BGE MODEL
    # ---------------------------------------------------------

    print("Loading embedding model...")
    print(MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    # ---------------------------------------------------------
    # CREATE EMBEDDINGS
    # ---------------------------------------------------------

    print()
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
    # 4-FOLD CROSS VALIDATION
    # ---------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    all_true = []
    all_pred = []

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(embeddings, labels),
        start=1,
    ):

        X_train = embeddings[train_idx]
        X_test = embeddings[test_idx]

        y_train = labels.iloc[train_idx].to_numpy()
        y_test = labels.iloc[test_idx].to_numpy()

        # Build intent prototypes ONLY from training data
        centroids = create_centroids(
            X_train,
            y_train,
        )

        predictions = predict_from_centroids(
            X_test,
            centroids,
        )

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
        all_pred.extend(predictions)

        print(
            f"Fold {fold}: "
            f"Accuracy={accuracy:.4f}, "
            f"Macro-F1={macro_f1:.4f}, "
            f"Weighted-F1={weighted_f1:.4f}"
        )

    # ---------------------------------------------------------
    # FINAL METRICS
    # ---------------------------------------------------------

    overall_accuracy = accuracy_score(
        all_true,
        all_pred,
    )

    overall_macro_f1 = f1_score(
        all_true,
        all_pred,
        labels=unique_labels,
        average="macro",
        zero_division=0,
    )

    overall_weighted_f1 = f1_score(
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

    print()
    print("=" * 70)
    print("SEMANTIC MODEL RESULTS")
    print("=" * 70)

    print(f"Accuracy:        {overall_accuracy:.4f}")
    print(f"Macro-F1:        {overall_macro_f1:.4f}")
    print(f"Weighted-F1:     {overall_weighted_f1:.4f}")

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    results = {
        "model": "BGE-small + nearest intent centroid",
        "embedding_model": MODEL_NAME,
        "dataset_size": len(df),
        "num_intents": len(unique_labels),
        "labels": unique_labels,
        "cross_validation": {
            "method": "4-fold stratified cross-validation",
            "random_state": RANDOM_STATE,
        },
        "folds": fold_results,
        "accuracy": float(overall_accuracy),
        "macro_f1": float(overall_macro_f1),
        "weighted_f1": float(overall_weighted_f1),
        "classification_report": report,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(f"Saved results → {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()