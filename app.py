import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import base64
ps = PorterStemmer()
st.set_page_config(
    page_title="SpamShield: Email & SMS Classifier",
    page_icon="📰"  # You can use an emoji or a custom image URL
)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    stopwords = nltk.corpus.stopwords.words('english')
    text = [word for word in text if word not in stopwords]
    text = [ps.stem(word) for word in text]
    text = ' '.join(text)
    return text

tfidf = pickle.load(open('Data/vectorizer.pkl','rb'))
model = pickle.load(open('Data/model.pkl','rb'))


# Function to set background
def set_background(local_image_path):
    with open(local_image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
set_background("Image.webp")  # Change to your image path

st.title("SpamShield: Email & SMS Classifier")
input_sms = st.text_area("Paste your email or SMS to check")

if st.button('Detect Spam'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")