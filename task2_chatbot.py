
import datetime
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

training_sentences = [
    "hello", "hi", "hey",
    "bye", "goodbye",
    "thanks", "thank you",
    "help", "what can you do",
    "who are you",
    "how are you"
]

intents = [
    "greeting", "greeting", "greeting",
    "bye", "bye",
    "thanks", "thanks",
    "help", "help",
    "identity",
    "status"
]

processed_training = [preprocess(s) for s in training_sentences]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_training)

model = MultinomialNB()
model.fit(X, intents)

print("Chatbot started! Type 'exit' to stop.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Chat ended.")
        break

    clean_input = preprocess(user_input)

    if "time" in clean_input:
        print("Bot:", datetime.datetime.now().strftime("%H:%M:%S"))
        continue

    if "date" in clean_input:
        print("Bot:", datetime.date.today())
        continue

    transformed = vectorizer.transform([clean_input])
    intent = model.predict(transformed)[0]

    if intent == "greeting":
        print("Bot: Hello! How can I help you?")

    elif intent == "bye":
        print("Bot: Goodbye! Have a nice day.")

    elif intent == "thanks":
        print("Bot: You're welcome!")

    elif intent == "help":
        print("Bot: I can chat, tell time/date, and answer simple questions.")

    elif intent == "identity":
        print("Bot: I am an AI chatbot created for Task-2 project.")

    elif intent == "status":
        print("Bot: I am working perfectly!")

    else:
        # Fallback response
        print("Bot: I don't understand. Please try again.")
