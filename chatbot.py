import json
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('wordnet')
nltk.download('omw-1.4')

class ChatBot:
    def __init__(self, intents_path):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.intents = self.load_intents(intents_path)
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
        self.train()

    def load_intents(self, path):
        with open(path, 'r') as file:
            return json.load(file)

    def preprocess(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word.lower()) for word in tokens]
        return " ".join(tokens)

    def train(self):
        corpus = []
        tags = []
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                processed = self.preprocess(pattern)
                corpus.append(processed)
                tags.append(intent['tag'])

        X = self.vectorizer.fit_transform(corpus)
        y = tags
        self.model.fit(X, y)

    def get_response(self, user_input):
        processed_input = self.preprocess(user_input)
        input_vector = self.vectorizer.transform([processed_input])
        prediction = self.model.predict(input_vector)[0]
        confidence = max(self.model.predict_proba(input_vector)[0])

        if confidence < 0.5:
            return "I'm not sure I understand. Can you rephrase that?"

        for intent in self.intents['intents']:
            if intent['tag'] == prediction:
                return random.choice(intent['responses'])

    def start_chat(self):
        print("🤖 ChatBot is ready! (type 'quit' to exit)\n")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                print("ChatBot: Goodbye! 👋")
                break
            response = self.get_response(user_input)
            print("ChatBot:", response)

# Run the bot
if __name__ == "__main__":
    chatbot = ChatBot("intents.json")
    chatbot.start_chat()