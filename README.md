# 🧠 AI Chatbot with NLP (NLTK + scikit-learn)

This is a simple AI chatbot built using **Python**, **NLTK**, and **scikit-learn**. It uses natural language processing to classify user input and respond based on predefined intents.

## 🚀 Features

- Intent classification using Logistic Regression
- Text preprocessing using NLTK
- Uses TF-IDF vectorizer for better accuracy
- Low-confidence fallback response
- Easy to extend with new intents

## 📁 Files

- `chatbot.py` — main chatbot script
- `intents.json` — training data with patterns and responses
- `requirements.txt` — required Python libraries

## ▶️ How to Run

```bash
pip install -r requirements.txt
python chatbot.py
```

## 💬 Sample Intents

- **greeting**: Hello, Hi, Hey
- **goodbye**: Bye, See you
- **thanks**: Thank you, Thanks
- **age**: How old are you?

## ✅ Example

```
You: Hello  
ChatBot: Hi there!  

You: How old are you?  
ChatBot: I was created recently!
```

---

### 🛠 Extend

You can edit `intents.json` to add more tags, patterns, and responses!

## 🧠 Built With

- [Python](https://www.python.org/)
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)

---

## 👩‍💻 Author

Vaishnavi Vibhute