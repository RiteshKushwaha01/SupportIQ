from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


INPUT_PATH = Path("evaluation/intent_annotation.csv")
OUTPUT_PATH = Path("evaluation/intent_classifier_metrics.json")

RANDOM_STATE = 42
N_SPLITS = 4


def evaluate_model(name, model, X, y, labels):
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []
    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        weighted_f1 = f1_score(
            y_test,
            predictions,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

        fold_results.append({
            "fold": fold,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })

        all_true.extend(y_test.tolist())
        all_pred.extend(predictions.tolist())

    overall_report = classification_report(
        all_true,
        all_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "model": name,
        "folds": fold_results,
        "mean_accuracy": float(
            np.mean([x["accuracy"] for x in fold_results])
        ),
        "mean_macro_f1": float(
            np.mean([x["macro_f1"] for x in fold_results])
        ),
        "mean_weighted_f1": float(
            np.mean([x["weighted_f1"] for x in fold_results])
        ),
        "classification_report": overall_report,
    }


def main():
    df = pd.read_csv(INPUT_PATH)

    df["text"] = df["text"].fillna("").astype(str)
    df["intent"] = df["intent"].fillna("").astype(str)

    df = df[df["intent"].str.strip() != ""].copy()

    X = df["text"]
    y = df["intent"]

    labels = sorted(y.unique())

    print("=" * 70)
    print("INTENT CLASSIFICATION EVALUATION")
    print("=" * 70)

    print(f"Examples: {len(df)}")
    print(f"Intents: {len(labels)}")
    print()

    print("Class distribution:")
    print(y.value_counts())
    print()

    # ---------------------------------------------------------
    # BASELINE 1: MAJORITY CLASS
    # ---------------------------------------------------------

    majority_class = y.value_counts().idxmax()

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    majority_true = []
    majority_pred = []

    for _, test_idx in skf.split(X, y):
        y_test = y.iloc[test_idx]

        majority_true.extend(y_test.tolist())
        majority_pred.extend(
            [majority_class] * len(test_idx)
        )

    majority_accuracy = accuracy_score(
        majority_true,
        majority_pred,
    )

    majority_macro_f1 = f1_score(
        majority_true,
        majority_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    majority_weighted_f1 = f1_score(
        majority_true,
        majority_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    majority_result = {
        "model": "majority_baseline",
        "majority_class": majority_class,
        "accuracy": float(majority_accuracy),
        "macro_f1": float(majority_macro_f1),
        "weighted_f1": float(majority_weighted_f1),
    }

    print("BASELINE 1 — Majority Class")
    print(f"Majority intent: {majority_class}")
    print(f"Accuracy:        {majority_accuracy:.4f}")
    print(f"Macro-F1:        {majority_macro_f1:.4f}")
    print(f"Weighted-F1:     {majority_weighted_f1:.4f}")
    print()

    # ---------------------------------------------------------
    # BASELINE 2: TF-IDF + LOGISTIC REGRESSION
    # ---------------------------------------------------------

    tfidf_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    tfidf_result = evaluate_model(
        "tfidf_logistic_regression",
        tfidf_model,
        X,
        y,
        labels,
    )

    print("BASELINE 2 — TF-IDF + Logistic Regression")
    print(
        f"Accuracy:        {tfidf_result['mean_accuracy']:.4f}"
    )
    print(
        f"Macro-F1:        {tfidf_result['mean_macro_f1']:.4f}"
    )
    print(
        f"Weighted-F1:     {tfidf_result['mean_weighted_f1']:.4f}"
    )
    print()

    # ---------------------------------------------------------
    # MODEL 3: TF-IDF + LINEAR SVM
    # ---------------------------------------------------------

    svm_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "classifier",
            LinearSVC(
                class_weight="balanced",
                C=1.0,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    svm_result = evaluate_model(
        "tfidf_linear_svm",
        svm_model,
        X,
        y,
        labels,
    )

    print("MODEL 3 — TF-IDF + Linear SVM")
    print(
        f"Accuracy:        {svm_result['mean_accuracy']:.4f}"
    )
    print(
        f"Macro-F1:        {svm_result['mean_macro_f1']:.4f}"
    )
    print(
        f"Weighted-F1:     {svm_result['mean_weighted_f1']:.4f}"
    )
    print()

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    results = {
        "dataset_size": len(df),
        "num_intents": len(labels),
        "labels": labels,
        "cross_validation": {
            "method": "5-fold stratified cross-validation",
            "random_state": RANDOM_STATE,
        },
        "class_distribution": {
            k: int(v)
            for k, v in y.value_counts().items()
        },
        "majority_baseline": majority_result,
        "tfidf_logistic_regression": tfidf_result,
         "tfidf_linear_svm": svm_result,
    }

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"Saved results → {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()