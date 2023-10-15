import streamlit as st

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np 
import json
import pickle

def load_json_data(file: str):
    with open(file, 'tr', encoding="UTF-8") as json_file:
        data = json.load(json_file)
    return data

def save_json_data(file: str, json_data):
    with open(file, 'tw', encoding="UTF-8") as outfile:
        json.dump(json_data, outfile, indent=4, ensure_ascii=False)

st.title("B. Prus - Lalka vol. 3")

with st.chat_message("assistant"):
    st.markdown("Chwileczkę, wczytuję model...")

tokenizer = Tokenizer()

data = load_json_data("all.json")
corpus = []
for d in data:
    corpus.append(d.lower())

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('max_sequence_len.pickle', 'rb') as handle:
    max_sequence_len = pickle.load(handle)

model = tf.keras.models.load_model('saved_model/lalka')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Ok. Jestem gotowy. Napisz początek zdania np. 'Nagle', a ja dokończę." +
        "Jestem trochę powolny, jak myślę, w prawym górnym rogu pokazuję 'RUNNING'."})


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Napisz początek zdania ..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    seed_text = prompt
    next_words = 13
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word

    response = seed_text
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})