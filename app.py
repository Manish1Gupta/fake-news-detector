"""
app.py
Streamlit web app for the Fake News Detector.
Run locally with: streamlit run app.py
"""

import streamlit as st
import pickle
import re
import string

# Load model + vectorizer
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detector")
st.write("Paste a news headline or article below, and the model will predict whether it's Real or Fake.")

user_input = st.text_area("News text", height=200, placeholder="Paste article or headline here...")

if st.button("Check News"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = round(max(proba) * 100, 2)

        if prediction == 1:
            st.success(f"✅ This looks like REAL news ({confidence}% confidence)")
        else:
            st.error(f"🚨 This looks like FAKE news ({confidence}% confidence)")

st.markdown("---")
st.caption("Built with Scikit-learn (TF-IDF + Logistic Regression) and Streamlit.")
