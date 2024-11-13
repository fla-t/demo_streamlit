import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np

import torch
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional


# Load the model and tokenizer
@st.cache_resource
def load_model() -> Tuple[AutoTokenizer, AutoModel]:
    model_name = "dmlls/all-mpnet-base-v2-negation"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


tokenizer, model = load_model()


@st.cache_data
def embed_texts(texts: List[str]) -> np.ndarray:
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings


def find_closest_match(
    sentence: str, keywords: List[str], max_distance: float = 0.5
) -> Tuple[Optional[str], Optional[float]]:
    # Embed the sentence and keywords
    sentence_embedding = embed_texts([sentence])[0]
    keyword_embeddings = embed_texts(keywords)

    # Compute cosine distances
    distances = cdist([sentence_embedding], keyword_embeddings, metric="cosine")[0]

    # Find the closest match within the threshold
    closest_idx = distances.argmin()
    closest_distance = distances[closest_idx]

    if closest_distance <= max_distance:
        return keywords[closest_idx], closest_distance
    else:
        return None, None  # Return None if no match is within the threshold


st.title("RTB-Search Embeddings Test with Pinecone")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_column = st.selectbox("Select the column containing sentences", df.columns)

    if text_column:
        sentences = df[text_column].fillna("").tolist()

        keyword_file = st.file_uploader(
            "Upload a text file with keywords (columns: Category, Keyword)", type="csv"
        )

        keywords = None

        if keyword_file:
            # combine the category and keyword columns into a single list
            keyword_df = pd.read_csv(keyword_file)
            keywords = keyword_df["keyword"].astype(str).tolist()

        if keywords and keywords[0].strip():
            # Slider for distance threshold
            max_distance = st.slider(
                "Set maximum distance threshold",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.001,
            )

            # Apply the function to each row
            df[["keyword_matched", "keyword_matched_distance"]] = df[text_column].apply(
                lambda sentence: pd.Series(
                    find_closest_match(sentence, [kw.strip() for kw in keywords], max_distance)
                )
            )

            st.write("Top keyword matches for each row:")
            st.dataframe(df)
            st.download_button(
                "Download results as CSV",
                df.to_csv(index=False),
                file_name="matched_keywords_with_distances.csv",
            )
