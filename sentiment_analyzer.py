import pandas as pd
import re
import string
import nltk
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from download_nltk import download_nltk_resources

download_nltk_resources()

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = LogisticRegression(max_iter=1000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))

        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and token.isalnum()]
        return ' '.join(tokens)

    def prepare_data(self, df):
        prepared_df = df.copy()
        label_map = {'Positive': 1, 'Negative': 0, 'Neutral': 2, 'Irrelevant': 3}
        prepared_df['Label'] = prepared_df['Label'].map(label_map)
        prepared_df = prepared_df.dropna(subset=['Label'])
        prepared_df['Label'] = prepared_df['Label'].astype(int)
        prepared_df['processed_text'] = prepared_df['Text'].apply(self.preprocess_text)
        return prepared_df[prepared_df['processed_text'] != ""]

    def train(self, X, y):
        X_vectorized = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_vectorized, y)

    def predict(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        vectorized_texts = self.vectorizer.transform(processed_texts)
        predictions = self.classifier.predict(vectorized_texts)
        label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral', 3: 'Irrelevant'}
        return [label_map[pred] for pred in predictions]
