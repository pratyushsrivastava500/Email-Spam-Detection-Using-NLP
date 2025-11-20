<<<<<<< HEAD
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
=======
"""
SpamShield: Email & SMS Spam Detector
A Streamlit web application for detecting spam in emails and SMS messages.
"""

import streamlit as st
import base64
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config.config import Config
from src.prediction import SpamPredictor

# Initialize configuration
config = Config()

# Set page configuration
st.set_page_config(
    page_title=config.page_title,
    page_icon=config.page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)


def set_background(local_image_path: str):
    """
    Set custom background image for the app.
    
    Args:
        local_image_path: Path to the background image
    """
    try:
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
    except FileNotFoundError:
        pass  # Silently ignore if background image not found


def load_predictor():
    """
    Load the spam predictor model.
    
    Returns:
        SpamPredictor instance or None if loading fails
    """
    try:
        # Use models directory for saved models
        model_path = config.models_dir / "model.pkl"
        vectorizer_path = config.models_dir / "vectorizer.pkl"
        scaler_path = config.models_dir / "scaler.pkl"
        
        # Fallback to Data directory if models not in models/
        if not model_path.exists():
            model_path = config.data_dir / "model.pkl"
            vectorizer_path = config.data_dir / "vectorizer.pkl"
            scaler_path = config.data_dir / "scaler.pkl"
        
        predictor = SpamPredictor(
            model_path=model_path,
            vectorizer_path=vectorizer_path,
            scaler_path=scaler_path if scaler_path.exists() else None
        )
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.warning("Please train the model first by running: `python train.py`")
        return None


def main():
    """Main application function."""
    
    # Set background (optional)
    if Path("images/Image.webp").exists():
        set_background("images/Image.webp")
    
    # Title and description
    st.title("🛡️ SpamShield: Email & SMS Classifier")
    st.markdown("""
    Welcome to **SpamShield**! This intelligent system uses Natural Language Processing (NLP) 
    and Machine Learning to detect spam in emails and SMS messages with high accuracy.
    """)
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **SpamShield** uses advanced NLP techniques including:
        - TF-IDF Vectorization
        - Text Preprocessing
        - Machine Learning Classification
        - Trained on thousands of messages
        """)
        
        st.header("📊 How to Use")
        st.markdown("""
        1. Paste your email or SMS text in the input box
        2. Click the **Detect Spam** button
        3. Get instant results with confidence scores
        """)
        
        st.header("🔒 Privacy")
        st.success("All processing is done locally. Your data is not stored or transmitted.")
    
    # Load the predictor
    predictor = load_predictor()
    
    if predictor is None:
        st.stop()
    
    # Display model information in expander
    with st.expander("🤖 Model Information"):
        model_info = predictor.get_model_info()
        st.json(model_info)
    
    # Main input area
    st.markdown("---")
    st.subheader("📝 Enter Message to Analyze")
    
    # Text input
    input_text = st.text_area(
        label="Paste your email or SMS message here:",
        height=200,
        placeholder="Enter the message you want to check for spam...",
        help="Enter any email or SMS text to check if it's spam"
    )
    
    # Create columns for buttons
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        detect_button = st.button('🔍 Detect Spam', type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button('🗑️ Clear', use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    # Process prediction
    if detect_button:
        if not input_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing message..."):
                try:
                    # Make prediction
                    prediction, label = predictor.predict(input_text)
                    
                    # Get probabilities if available
                    try:
                        ham_prob, spam_prob = predictor.predict_proba(input_text)
                    except:
                        ham_prob, spam_prob = (1.0, 0.0) if prediction == 0 else (0.0, 1.0)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Analysis Results")
                    
                    # Create result columns
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        if label == "Spam":
                            st.error("### 🚫 SPAM DETECTED")
                            st.markdown("""
                            **This message appears to be spam.**
                            - Be cautious with links or attachments
                            - Do not share personal information
                            - Consider deleting this message
                            """)
                        else:
                            st.success("### ✅ NOT SPAM")
                            st.markdown("""
                            **This message appears to be legitimate (Ham).**
                            - The message seems safe
                            - Always verify sender identity
                            - Stay vigilant with unfamiliar senders
                            """)
                    
                    with result_col2:
                        st.markdown("### 📊 Confidence Scores")
                        
                        # Display probability bars
                        st.metric("Ham Probability", f"{ham_prob*100:.2f}%")
                        st.progress(ham_prob)
                        
                        st.metric("Spam Probability", f"{spam_prob*100:.2f}%")
                        st.progress(spam_prob)
                    
                    # Show preprocessed text in expander
                    with st.expander("🔍 View Preprocessed Text"):
                        preprocessed = predictor.preprocess_text(input_text)
                        st.code(preprocessed, language="text")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Made with ❤️ using Streamlit and Machine Learning | © 2025 SpamShield</p>
        <p>Powered by Natural Language Processing and Python</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
>>>>>>> 64048f2 (Updated Code)
