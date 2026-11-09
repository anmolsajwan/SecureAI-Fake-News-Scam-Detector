
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("data/fake_news.csv")

# Remove missing values
df = df.dropna(subset=["title", "text", "label"])

# Combine title and news text
df["content"] = df["title"] + " " + df["text"]

# Clean news text
df["content"] = df["content"].apply(clean_text)

# Input and output
X = df["content"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
print("Training Fake News Model...")
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, "models/fake_news_model.pkl")

print("\nFake News Model saved successfully!")
