from pathlib import Path

import faiss
import sqlite3
import numpy as np
from fastembed import TextEmbedding

class Retriever:
    def __init__(
        self,
        corpus_path,
        embeddings_path,
        index_path,
        model_name="BAAI/bge-small-en-v1.5",
        model=None,
    ):
        self.corpus_path = Path(corpus_path)
        self.embeddings_path = Path(embeddings_path)
        self.index_path = Path(index_path)

        self.db_path = self.corpus_path.with_suffix(".db")
        if not self.db_path.exists():
            raise FileNotFoundError(f"Retrieval database not found: {self.db_path}")
        print("Opening lightweight retrieval database...")
        self.db = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)

        print(
            "Using FAISS index for retrieval; "
            "precomputed embeddings are not loaded at inference time."
        )

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(self.index_path))

        print("Loading embedding model...")
        self.model = model if model is not None else TextEmbedding(model_name=model_name)

        print("Retriever ready!")
        print("Documents:", self.index.ntotal)
        print("FAISS vectors:", self.index.ntotal)

    def search(self, query, top_k=5):
        query_embedding = np.array(
            list(self.model.embed([query]))[0],
            dtype="float32",
        ).reshape(1, -1)

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
            row = self.db.execute("SELECT customer_tweet_id, conversation_id, customer_query, agent_response FROM documents WHERE rowid = ?", (int(idx) + 1,)).fetchone()
            if row is None:
                continue
            customer_tweet_id, conversation_id, customer_query, agent_response = row
            results.append({
                "rank": rank,
                "similarity": float(score),
                "customer_query": customer_query,
                "agent_response": agent_response,
                "conversation_id": conversation_id,
                "customer_tweet_id": customer_tweet_id,
            })

        return results