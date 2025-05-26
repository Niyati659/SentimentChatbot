import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens
                  if token not in self.stop_words and token.isalnum()]
        return ' '.join(tokens)

    def prepare_data(self, df):
        prepared_df = df.copy()
        label_map = {'Positive': 1, 'Negative': 0, 'Neutral': 2, 'Irrelevant': 3}
        prepared_df['Label'] = prepared_df['Label'].apply(lambda x: label_map.get(x, x))
        prepared_df = prepared_df[pd.to_numeric(prepared_df['Label'], errors='coerce').notnull()]
        prepared_df['Label'] = prepared_df['Label'].astype(int)
        prepared_df['processed_text'] = prepared_df['Text'].apply(self.preprocess_text)
        prepared_df = prepared_df[prepared_df['processed_text'] != ""]
        return prepared_df

    def train(self, X_train, y_train):
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vectorized, y_train)

    def predict(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        vectorized_texts = self.vectorizer.transform(processed_texts)
        predictions = self.classifier.predict(vectorized_texts)
        label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral', 3: 'Irrelevant'}
        return [label_map[pred] for pred in predictions]
