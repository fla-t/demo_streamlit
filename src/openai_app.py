from typing import List
from io import BytesIO

import streamlit as st
import pandas as pd
from openai import OpenAI
from scipy import spatial


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[List[float]]:

    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [distance_metrics[distance_metric](query_embedding, embeddings)]
    return distances[0]


client = OpenAI()

st.write(""" # RTB-Search Embeddings Test with OpenAI """)

input = st.text_input("Search")
file = st.file_uploader(
    label='Upload a csv file, the must contain a "notes" column',
    type=["csv"],
    accept_multiple_files=False,
)


def get_embedding(text: str, model="text-embedding-3-small"):
    if len(text) == 0:
        return None
    print("START", text, "END")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def search(df: pd.DataFrame, search_key: str, n: int = 3):
    query_embedding = get_embedding(search_key, model="text-embedding-3-small")
    df["similarities"] = df.ada_embedding.apply(
        lambda x: distances_from_embeddings(query_embedding, x, distance_metric="cosine")
    )
    res = df.sort_values("similarities", ascending=True).head(n)
    return res


if file and input:
    df = pd.read_csv(BytesIO(file.read()))
    df["ada_embedding"] = df["note"].apply(
        lambda x: get_embedding(x, model="text-embedding-3-small")
    )
    df.dropna(subset=["note"], inplace=True)

    keywords = input.split(",")

    res = search(df, input, n=10)
    matches = "\n".join(res.note.to_list())
    st.text_area("Results", matches)

    st.download_button(
        label="Download the original csv file",
        data=df.to_csv(index=False).encode(),
        file_name="original.csv",
        mime="text/csv",
    )
