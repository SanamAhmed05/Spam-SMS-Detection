import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

st.set_page_config(page_title="SMS Spam Detection", page_icon="📩")

st.title("📩 SMS Spam Detection")

sms = st.text_area("Enter your SMS")

if st.button("Predict"):

    transformed_sms = transform_text(sms)

    vector = tfidf.transform([transformed_sms])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")
