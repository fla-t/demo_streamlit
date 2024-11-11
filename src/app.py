import streamlit as st
import pandas as pd
from io import BytesIO

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

    try:
        pinecone.find_and_save_embeddings(data=data, index="example", namespace=namespace)
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
    finally:
        st.success("Embeddings saved successfully!")

    input = st.text_input("Search (Add multiple terms separated with comma)", key="search")
    results = pinecone.query_index(
        query=input.split(","),
        index="example",
        namespace=namespace,
        include_values=False,
        include_metadata=True,
    )

    print(notes)
    st.download_button(
        label="Download the original csv file",
        data=notes.to_csv(index=False).encode(),
        file_name="original.csv",
        mime="text/csv",
    )

    if results:
        st.text(str(results))
