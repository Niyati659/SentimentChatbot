import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sentiment_analyzer import SentimentAnalyzer

@st.cache_resource
def load_analyzer():
    analyzer = SentimentAnalyzer()
    df = pd.read_csv("twitter_training.csv", encoding="latin-1", header=None)
    df.columns = ['ID', 'Entity', 'Label', 'Text']
    new_df = df[["Label", "Text"]].dropna()
    prepared_df = analyzer.prepare_data(new_df)
    X_train, _, y_train, _ = train_test_split(
        prepared_df['processed_text'], prepared_df['Label'], test_size=0.2, random_state=42, stratify=prepared_df['Label']
    )
    analyzer.train(X_train, y_train)
    return analyzer

st.title("🧠 Sentiment Analysis Chatbot")

analyzer = load_analyzer()

text_input = st.text_input("Enter a sentence to analyze:")

if text_input:
    prediction = analyzer.predict([text_input])[0]
    st.success(f"Predicted Sentiment: **{prediction}**")
