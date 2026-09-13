from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        corpus_path,
        embeddings_path,
        index_path,
        model_name="BAAI/bge-small-en-v1.5",
    ):
        self.corpus_path = Path(corpus_path)
        self.embeddings_path = Path(embeddings_path)
        self.index_path = Path(index_path)

        print("Loading retrieval corpus...")
        self.documents = pd.read_parquet(self.corpus_path)

        print("Loading embeddings...")
        self.embeddings = np.load(self.embeddings_path)

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print("Retriever ready!")
        print("Documents:", len(self.documents))
        print("Embeddings:", self.embeddings.shape)
        print("FAISS vectors:", self.index.ntotal)

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):
            row = self.documents.iloc[idx]

            results.append({
                "rank": rank,
                "similarity": float(score),
                "customer_query": row["customer_query"],
                "agent_response": row["agent_response"],
                "conversation_id": row["conversation_id"],
                "customer_tweet_id": row["customer_tweet_id"],
            })

        return results