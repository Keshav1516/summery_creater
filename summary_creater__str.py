pip install transformers sentence_transformers textstat streamlit

import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import textstat

def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return summarizer, sbert_model

summarizer, sbert_model = load_models()

st.title("AI Text Summarizer, Readability Checker & Semantic Matcher")

# Input Text
input_text = st.text_area("Enter the content to summarize and check readability:", height=200)

# Query for similarity comparison
query = st.text_input("Enter a query to check semantic similarity with the content:")

# Process on button click
if st.button("Generate Summary & Analyze"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Summarization
        summary_result = summarizer(input_text, max_length=60, min_length=30, do_sample=False)
        summary_text = summary_result[0]['summary_text']
        st.subheader("Summary:")
        st.write(summary_text)

        # Readability
        readability_score = textstat.flesch_reading_ease(input_text)
        st.subheader("Readability Score (Flesch Reading Ease):")
        st.write(f"{readability_score:.2f}")

        # Semantic similarity
        if query.strip():
            emb1 = sbert_model.encode(query, convert_to_tensor=True)
            emb2 = sbert_model.encode(input_text, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2).item()
            st.subheader("Semantic Similarity Score:")
            st.write(f"{similarity:.4f}")
        else:
            st.info("No query entered, skipping similarity comparison.")

