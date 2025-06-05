import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
import pickle

# Initialize the stemmer
stemmer = PorterStemmer()

# Load the intents data
with open('intents.json') as file:
    data = json.load(file)

# Prepare data: patterns and tags
patterns = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Function to clean and stem each sentence
def stem_sentence(sentence):
    tokens = wordpunct_tokenize(sentence.lower())  # More reliable tokenizer
    return ' '.join([stemmer.stem(w) for w in tokens if w.isalpha()])

# Apply stemming to all patterns
stemmed_patterns = [stem_sentence(p) for p in patterns]

# Convert text to numerical form using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(stemmed_patterns)

# The labels are the tags
y = tags

# Train the machine learning model
model = LogisticRegression()
model.fit(X, y)

# Save the model and vectorizer for later use
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved as 'model.pkl' and 'vectorizer.pkl'")
