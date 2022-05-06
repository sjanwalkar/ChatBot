#!/usr/bin/env python
# coding: utf-8

# # Chatbot Flask Server

# In[1]:


# Imports
import flask
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import json
import random
import numpy as np
import nltk
import pickle
from nltk.stem import PorterStemmer


# In[2]:


app = Flask(__name__)

# Reading Saved Files
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
all_words = pickle.load(open('all_words.pkl','rb'))
tags = pickle.load(open('tags.pkl','rb'))


# In[3]:


def tokenize(sentence):
    return nltk.word_tokenize(sentence)
stemmer = PorterStemmer()
def stem(word):
    return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence ]
    bag = np.zeros(len(all_words), dtype= np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


# In[4]:


def user_input(sentence):    
    # Tokenization
    sentence = tokenize(sentence)
    # lowering and stemming
    # Generating Bag of Words
    bow = bag_of_words(sentence, all_words)
    return np.array(bow)

THRESHOLD=0.25
def response(sentence):
    p = user_input(sentence)
    results = model.predict(np.array([p]))[0]
  # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>THRESHOLD]
  # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((tags[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def chatbot(sentence):
    result = response(sentence)
    if result:
        while result:

            for i in intents['intents']:
      # find a tag matching the first result
                if i['tag'] == result[0][0]:
        # a random response from the intent
                    return (random.choice(i['responses']))


# In[ ]:


app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot(userText)
if __name__ == "__main__":
    app.run()

