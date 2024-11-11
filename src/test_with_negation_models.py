from sentence_transformers import SentenceTransformer

sentences = [
    "I like rainy days because they make me feel relaxed.",
    "I don't like rainy days because they don't make me feel relaxed.",
]

model = SentenceTransformer("dmlls/all-mpnet-base-v2-negation")
embeddings = model.encode(sentences)
print(embeddings)
