from typing import Any
import numpy as np

from pinecone.grpc import PineconeGRPC as Pinecone


class Emebdding:
    text: str
    embedding: np.ndarray[Any, np.dtype[np.float64]]

    def __init__(self, text: str, embedding: list[float]) -> None:
        self.text = text
        self.embedding = np.array(embedding)

    def __str__(self):
        return f"Text: {self.text}, Embedding: {self.embedding[:10]}..."


class PineconeWrapper:

    def __init__(self, api_key: str):
        self.pc = Pinecone(api_key=api_key)

    def find_embeddings(self, data: list[str]) -> list[Emebdding]:
        embeddings = self.pc.inference.embed(
            model="multilingual-e5-large",
            inputs=data,
            parameters={"input_type": "passage", "truncate": "END"},
        )

        return [Emebdding(text=d, embedding=e["values"]) for d, e in zip(data, embeddings)]

    def find_and_save_embeddings(
        self,
        data: list[str],
        index: str,
        namespace: str,
    ):
        embeddings = self.pc.inference.embed(
            model="multilingual-e5-large",
            inputs=data,
            parameters={"input_type": "passage", "truncate": "END"},
        )

        index = self.pc.Index(index)
        records = []

        for d, e, idx in zip(data, embeddings, range(len(data))):
            records.append({"id": str(idx + 1), "values": e["values"], "metadata": {"text": d}})

        index.upsert(vectors=records, namespace=namespace)

    def query_index(
        self,
        query: list[str],
        index: str,
        namespace: str,
        include_values: bool = False,
        include_metadata: bool = False,
        confidence_threshold: float = 0.9,
    ) -> Any:
        query_embedding = self.pc.inference.embed(
            model="multilingual-e5-large", inputs=query, parameters={"input_type": "query"}
        )

        index = self.pc.Index(index)
        results = index.query(
            namespace=namespace,
            vector=query_embedding[0].values,
            top_k=10,
            include_values=include_values,
            include_metadata=include_metadata,
        )

        return results
