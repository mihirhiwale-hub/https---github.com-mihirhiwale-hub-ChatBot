from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import datetime

training_sentences = [
    "hello", "hi", "hey",
    "bye", "goodbye",
    "thanks", "thank you",
    "help", "what can you do",
    "how are you",
    "who are you"
]

intents = [
    "greeting", "greeting", "greeting",
    "bye", "bye",
    "thanks", "thanks",
    "help", "help",
    "status",
    "identity"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

model = MultinomialNB()
model.fit(X, intents)

print("Chatbot started! Type 'exit' to stop.")

while True:
    user_input = input("You: ").lower()

    if user_input == "exit":
        print("Bot: Chat ended.")
        break

    # RULE-BASED PART (Fix)
    if "time" in user_input:
        print("Bot:", datetime.datetime.now().strftime("%H:%M:%S"))
        continue

    if "date" in user_input:
        print("Bot:", datetime.date.today())
        continue

    # ML PART
    transformed = vectorizer.transform([user_input])
    intent = model.predict(transformed)[0]

    if intent == "greeting":
        print("Bot: Hello! How can I help you?")

    elif intent == "bye":
        print("Bot: Goodbye! Have a nice day.")

    elif intent == "thanks":
        print("Bot: You're welcome!")

    elif intent == "help":
        print("Bot: I can chat, tell time/date, and answer simple questions.")

    elif intent == "status":
        print("Bot: I am working perfectly!")

    elif intent == "identity":
        print("Bot: I am an AI chatbot created for Task-2 project.")

    else:
        print("Bot: I don't understand.")
