import csv
import json
import os

import requests
import streamlit as st
from annoy import AnnoyIndex

st.set_page_config(layout="wide")

STORAGE = os.getenv("STORAGE")
ANNOY_INDEX_FILE = "card-embeddings.ann"
EMBEDDING_SIZE = 50


@st.cache_resource
def load_vector_database():
    import requests

    url = f"{STORAGE}/card-embeddings.ann"
    r = requests.get(url)
    with open(ANNOY_INDEX_FILE, "wb") as f:
        f.write(r.content)

    ann = AnnoyIndex(EMBEDDING_SIZE, "angular")
    ann.load(ANNOY_INDEX_FILE)

    return ann


@st.cache_resource
def load_cards():
    import requests

    url = f"{STORAGE}/cards.csv"
    r = requests.get(url)

    with open("cards.csv", "wb") as f:
        f.write(r.content)

    cards = {}
    with open("cards.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cards[row["id"]] = row

    return cards


@st.cache_resource
def load_supporting_dictionsaries():
    url_passcode_to_id = f"{STORAGE}/passcode_to_id.json"
    r = requests.get(url_passcode_to_id)
    with open("passcode_to_id.json", "wb") as f:
        f.write(r.content)

    passcode_to_id = {}
    with open("passcode_to_id.json", "r") as f:
        passcode_to_id = json.load(f)

    url_id_to_passcode = f"{STORAGE}/id_to_passcode.json"
    r = requests.get(url_id_to_passcode)
    with open("id_to_passcode.json", "wb") as f:
        f.write(r.content)

    id_to_passcode = {}
    with open("id_to_passcode.json", "r") as f:
        id_to_passcode = json.load(f)

    return passcode_to_id, id_to_passcode


if STORAGE is None:
    st.error("Please set the STORAGE environment variable to the URL of the storage bucket.")
else:
    ann = load_vector_database()
    cards = load_cards()
    passcode_to_id, id_to_passcode = load_supporting_dictionsaries()

    def format_card(card_id):
        return cards[card_id]["name"]

    st.title("Yu-Gi-Oh! card recommender")
    st.write("Welcome to the recommender! Please select a card to get started.")

    query_card_passcode = st.selectbox("Select a card", cards.keys(), format_func=format_card)

    card_id = passcode_to_id[query_card_passcode]
    query_card_embedding = ann.get_item_vector(card_id)

    similar_card_ids, scores = ann.get_nns_by_vector(query_card_embedding, 6, include_distances=True)

    # Check if the query card is in the list of similar cards
    if similar_card_ids[0] == card_id:
        similar_card_ids.pop(0)
        scores.pop(0)

    # Make sure we limit to 5 similar cards
    similar_card_ids = similar_card_ids[:5]
    scores = scores[:5]

    st.write("Here are some similar cards:")
    columns = st.columns(len(similar_card_ids) + 1)

    with columns[0]:
        # passcode = id_to_passcode[str(similar_card_ids[0])]
        st.subheader("Query Card:")
        st.image(cards[query_card_passcode]["image_url"])

    for similar_card_id, score, column in zip(similar_card_ids, scores, columns[1:]):
        with column:
            passcode = id_to_passcode[str(similar_card_id)]
            st.subheader(f"Distance {score:.3f}")
            st.image(cards[passcode]["image_url"])