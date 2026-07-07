"""
train_model.py
Trains a Fake News classifier and saves the model + vectorizer to disk.

BEFORE RUNNING:
1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Place Fake.csv and True.csv inside the "data" folder in this project.
"""

import pandas as pd
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load data
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0  # 0 = Fake
true["label"] = 1  # 1 = Real

data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# Combine title + text for a stronger signal
data["content"] = data["title"] + " " + data["text"]


# 2. Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


data["content"] = data["content"].apply(clean_text)


# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["content"], data["label"], test_size=0.2, random_state=42
)


# 4. TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 5. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


# 6. Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 7. Save model + vectorizer
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved to /model folder.")
