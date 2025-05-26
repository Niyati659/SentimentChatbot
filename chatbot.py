import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("SentimentalAnaylser/twitter_training.csv", encoding="latin-1", header=None)
df.columns = ['ID', 'Entity', 'Label', 'Text']
print(df.head())
new_df = df[["Label", "Text"]].dropna()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)
nltk.download('wordnet', force=True)
nltk.download('omw-1.4', force=True)

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

        try:
            tokens = word_tokenize(text)
        except LookupError:
            nltk.download('punkt')
            tokens = word_tokenize(text)

        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and token.isalnum()]
        return ' '.join(tokens)

    def prepare_data(self, df):
        prepared_df = df.copy()

        print("Unique labels before mapping:", prepared_df['Label'].unique())

        label_map = {'Positive': 1, 'Negative': 0, 'Neutral': 2, 'Irrelevant': 3}
        prepared_df['Label'] = prepared_df['Label'].apply(lambda x: label_map.get(x, x))
        prepared_df = prepared_df[pd.to_numeric(prepared_df['Label'], errors='coerce').notnull()]
        prepared_df['Label'] = prepared_df['Label'].astype(int)

        print("Preprocessing texts...")
        prepared_df['processed_text'] = prepared_df['Text'].apply(self.preprocess_text)
        prepared_df = prepared_df[prepared_df['processed_text'] != ""]

        return prepared_df

    def train(self, X_train, y_train):
        print("Vectorizing text...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)

        print("Training classifier...")
        self.classifier.fit(X_train_vectorized, y_train)

    def evaluate(self, X_test, y_test):
        X_test_vectorized = self.vectorizer.transform(X_test)
        predictions = self.classifier.predict(X_test_vectorized)

        print("\nClassification Report:")
        print(classification_report(
            y_test, predictions, target_names=['Negative', 'Positive', 'Neutral', 'Irrelevant']))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        return predictions

    def predict(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        vectorized_texts = self.vectorizer.transform(processed_texts)
        predictions = self.classifier.predict(vectorized_texts)
        label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral', 3: 'Irrelevant'}
        return [label_map[pred] for pred in predictions]

def main():
    analyzer = SentimentAnalyzer()
    
    print("Original Label values in dataset:", new_df['Label'].unique())
    
    prepared_df = analyzer.prepare_data(new_df)

    print("\nLabel distribution after preprocessing:")
    print(prepared_df['Label'].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        prepared_df['processed_text'],
        prepared_df['Label'],
        test_size=0.2,
        random_state=42,
        stratify=prepared_df['Label']
    )

    analyzer.train(X_train, y_train)
    analyzer.evaluate(X_test, y_test)

    new_texts = [
        "This is absolutely amazing!",
        "I'm not sure how I feel about this",
        "This is the worst experience ever"
    ]

    print("\nPredictions for new texts:")
    predictions = analyzer.predict(new_texts)
    for text, pred in zip(new_texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {pred}\n")

if __name__ == "__main__":
    main()
