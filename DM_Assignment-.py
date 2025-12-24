import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Twitter Sentiment Analyzer")

model = load_model("cnn_sentiment.h5")
tokenizer = pickle.load(open("tokenizer.pkl","rb"))

text = st.text_area("Enter Tweet")

if st.button("Predict"):
    clean = text.lower()
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=100)
    pred = model.predict(pad)[0][0]

    if pred > 0.5:
        st.success("Positive 😊")
    else:
        st.error("Negative 😞")
