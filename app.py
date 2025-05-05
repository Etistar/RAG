import streamlit as st
from answer_module import answer  # si tu mets ton code dans answer_module.py

st.set_page_config(page_title="Assistant ILM", layout="centered")

st.title("🧠 Assistant ILM")

query = st.text_input("Pose ta question :", "")

if query:
    with st.spinner("Recherche de la réponse..."):
        response = answer(query)
        st.markdown(response)
