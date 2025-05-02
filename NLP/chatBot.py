# chatBot.py

import nltk
import warnings
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings
warnings.filterwarnings("ignore")

# Download NLTK data (ensure it's done once)
nltk.download('punkt')
nltk.download('wordnet')

# Load and preprocess text files
with open('answer.txt', 'r', errors='ignore') as f:
    raw = f.read().lower()
with open('chatbot.txt', 'r', errors='ignore') as m:
    rawone = m.read().lower()

# Tokenize
sent_tokens = nltk.sent_tokenize(raw)
sent_tokensone = nltk.sent_tokenize(rawone)

# Lemmatization setup
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Predefined responses
Introduce_Ans = [
    "My name is Meteor Bot.",
    "You can call me Meteor Bot or B.O.T.",
    "I'm PyBot, happy to chat!",
]
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "hello", "hi there", "hello there"]
Basic_Q = ("what is python", "what is python?")
Basic_Ans = "Python is a high-level, interpreted programming language."
Basic_Om = (
    "what is module", "what is module?", "what is module in python", "what is module in python?"
)
Basic_AnsM = [
    "A module is a file containing Python code, like functions and classes.",
    "Modules help organize and reuse code.",
    "Think of a module as a toolbox for Python functions."
]

# Response helpers
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def basic(sentence):
    if sentence.lower() in Basic_Q:
        return Basic_Ans

def basicM(sentence):
    if sentence.lower() in Basic_Om:
        return random.choice(Basic_AnsM)

def IntroduceMe(sentence):
    return random.choice(Introduce_Ans)

# TF-IDF response generator
def generate_response(user_response, corpus):
    robo_response = ''
    corpus.append(user_response)
    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words=None)
    tfidf = vectorizer.fit_transform(corpus)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response += "I'm sorry, I didn't understand that."
    else:
        robo_response += corpus[idx]
    corpus.pop()  # Remove user input
    return robo_response

# Main chat interface
def chat(user_response):
    user_response = user_response.lower()

    if user_response == 'bye':
        return "Bye! Take care."

    if user_response in ['thanks', 'thank you']:
        return "You're welcome."

    if greeting(user_response):
        return greeting(user_response)

    if "your name" in user_response:
        return IntroduceMe(user_response)

    if basic(user_response):
        return basic(user_response)

    if basicM(user_response):
        return basicM(user_response)

    if "module" in user_response:
        return generate_response(user_response, sent_tokensone)

    return generate_response(user_response, sent_tokens)
