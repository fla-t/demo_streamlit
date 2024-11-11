import streamlit as st
import pandas as pd
from io import BytesIO
import numpy as np

from utils import PineconeWrapper

pinecone = PineconeWrapper(api_key="38d7c4af-7635-403a-9c87-f9914037b770")
st.write(""" # RTB-Search Embeddings Test with Pinecone """)

file = st.file_uploader(
    label='Upload a csv file, the must contain a "notes" column',
    type=["csv"],
    accept_multiple_files=False,
)

if file:
    notes = pd.read_csv(BytesIO(file.read()))
    data = notes["note"].astype(str).tolist()
    namespace = "example-some-namespace"

    search_keyword_embeddings = None
    embeddings = None

    try:
        embeddings = pinecone.find_embeddings(data=data)
        st.success("Embeddings fetched successfully!")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    input = st.text_input("Search (Add multiple terms separated with comma)", key="search")

    try:
        if input.strip() != "":
            search_keyword_embeddings = pinecone.find_embeddings(data=input.split(","))
        st.success("Embeddings fetched successfully!")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    if search_keyword_embeddings and embeddings:
        st.text(str(search_keyword_embeddings))
        st.text(str(embeddings))

        # find cosine similarity between search_keyword_embeddings and embeddings, all possible combinations
        cosine_similarities = []
        for search_keyword_embedding in search_keyword_embeddings:
            for embedding in embeddings:
                cosine_similarities.append(
                    [
                        np.dot(search_keyword_embedding.embedding, embedding.embedding)
                        / (
                            np.linalg.norm(search_keyword_embedding.embedding)
                            * np.linalg.norm(embedding.embedding)
                        ),
                        search_keyword_embedding.text,
                        embedding.text,
                    ]
                )

        cosine_similarities.sort(key=lambda x: x[0], reverse=True)
        st.text(str(cosine_similarities))
        print(cosine_similarities)
