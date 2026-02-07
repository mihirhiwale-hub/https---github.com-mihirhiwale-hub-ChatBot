import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents
with open("data/intents.json") as f:
    intents = json.load(f)["intents"]

# Prepare training data
patterns = []
tags = []
responses = {}

for intent in intents:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
model = LogisticRegression()
model.fit(X, tags)

def get_response(user_input):
    try:
        X_test = vectorizer.transform([user_input])
        predicted_tag = model.predict(X_test)[0]
        return random.choice(responses[predicted_tag])
    except:
        return random.choice(responses["fallback"])