from pathlib import Path

import pandas as pd

from src.intent_classifier import IntentClassifier


ANNOTATION_PATH = Path(
    "evaluation/intent_annotation.csv"
)


def load_intent_classifier():

    df = pd.read_csv(ANNOTATION_PATH)

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    df["intent"] = (
        df["intent"]
        .fillna("")
        .astype(str)
    )

    df = df[
        df["intent"].str.strip() != ""
    ].copy()

    classifier = IntentClassifier()

    classifier.fit(
        df["text"].tolist(),
        df["intent"].tolist(),
    )

    return classifier