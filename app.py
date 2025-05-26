import streamlit as st
import pandas as pd
from sentiment_analyzer import SentimentAnalyzer
import os
import nltk
nltk.data.path.append("nltk_data") 
import nltk
print(nltk.data.find("tokenizers/punkt")) # ensure NLTK uses your bundled data
@st.cache_resource
def load_analyzer():
    analyzer = SentimentAnalyzer()
    df = pd.read_csv("twitter_training.csv", encoding="latin-1", header=None)
    df.columns = ['ID', 'Entity', 'Label', 'Text']
    new_df = df[["Label", "Text"]].dropna()
    prepared_df = analyzer.prepare_data(new_df)

    X = prepared_df['processed_text']
    y = prepared_df['Label']
    analyzer.train(X, y)
    return analyzer

st.title("💬 Sentiment Chatbot")

analyzer = load_analyzer()

user_input = st.text_input("Enter your message:")

if user_input:
    prediction = analyzer.predict([user_input])[0]
    st.markdown(f"**Sentiment:** `{prediction}`")
