import random
import json
import nltk
import numpy as np
import pickle
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer

# Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load intents
with open("intents.json") as f:
    intents = json.load(f)

# Initialize stemmer
stemmer = PorterStemmer()

# Preprocessing function
def preprocess(sentence):
    tokens = wordpunct_tokenize(sentence.lower())
    return ' '.join([stemmer.stem(w) for w in tokens if w.isalpha()])

# Predict tag
def predict_tag(msg):
    msg_processed = preprocess(msg)
    X_test = vectorizer.transform([msg_processed])
    return model.predict(X_test)[0]

# Get response
def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# Chat loop
print("🤖 Chatbot is ready! Type 'quit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bot: Goodbye! 👋")
        break

    tag = predict_tag(user_input)
    response = get_response(tag)
    print("Bot:", response)
