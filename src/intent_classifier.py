import numpy as np
from fastembed import TextEmbedding


class IntentClassifier:
    """
    Semantic intent classifier using BGE embeddings
    and intent centroids.
    """

    INTENTS = [
        "delivery_tracking",
        "order_management",
        "returns_refunds",
        "payment_billing",
        "product_information",
        "technical_support",
        "account_subscription",
        "seller_marketplace",
        "other",
    ]

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", model=None):
        self.model = model if model is not None else TextEmbedding(model_name=model_name)
        self.centroids = None

    def fit(self, texts, labels):
        """
        Build one semantic centroid for every intent.
        """

        embeddings = np.array(
            list(self.model.embed(list(texts))),
            dtype="float32",
        )

        self.centroids = {}

        for intent in self.INTENTS:
            mask = np.array(labels) == intent

            if not np.any(mask):
                continue

            centroid = embeddings[mask].mean(axis=0)

            norm = np.linalg.norm(centroid)

            if norm > 0:
                centroid = centroid / norm

            self.centroids[intent] = centroid

        return self

    def predict(self, text):
        """
        Predict the most semantically similar intent.
        """

        if not self.centroids:
            raise RuntimeError(
                "Classifier has not been fitted."
            )

        embedding = np.array(
            list(self.model.embed([text]))[0],
            dtype="float32",
        )

        intents = list(self.centroids.keys())

        centroid_matrix = np.vstack(
            [self.centroids[intent] for intent in intents]
        )

        similarities = centroid_matrix @ embedding

        best_index = int(np.argmax(similarities))

        intent = intents[best_index]
        confidence = float(similarities[best_index])

        return {
            "intent": intent,
            "semantic_score": confidence
        }