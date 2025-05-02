import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib  # For loading the saved model
import re
import time

# Function to count hashtags
def count_hashtags(tweet):
    hashtags = re.findall(r'\#\w+', tweet)  # Find all hashtags
    return len(hashtags)

# Load pre-trained model and TF-IDF vectorizer
model = joblib.load('logistic_regression_model.pkl')  # Save the model using joblib
tfidf = joblib.load('tfidf_vectorizer.pkl')  # Save the TF-IDF vectorizer using joblib

# Streamlit UI Setup
st.set_page_config(page_title="Tweet Virality Prediction", page_icon="🚀", layout="centered")

# App Title and Subtitle
st.markdown("<h1 style='text-align: center; font-size: 36px; font-weight: bold;'>🚀 Tweet Virality Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>Enter a tweet and predict its virality in seconds!</h3>", unsafe_allow_html=True)

# Main content container
with st.container():
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

    # Input section: Tweet content with placeholder text
    tweet = st.text_area("", "", height=200, max_chars=500, placeholder="Type your tweet here...", label_visibility="collapsed")

    # Button for Prediction
    if st.button("Predict Virality", help="Click to predict whether the tweet is viral or not"):
        if tweet:
            # Show a loading spinner while processing
            with st.spinner('Analyzing Tweet...'):
                time.sleep(1)  # Simulating processing time
                num_hashtags = count_hashtags(tweet)

                # Transform tweet using the TF-IDF vectorizer
                tweet_tfidf = tfidf.transform([tweet]).toarray()

                # Combine the TF-IDF vector and the number of hashtags as input for prediction
                X_input = np.hstack((tweet_tfidf, np.array([[num_hashtags]])))

                # Make a prediction
                prediction = model.predict(X_input)

                # Display prediction result with styling
                if prediction[0] == 1:
                    st.markdown("<div style='background-color: #c8e6c9; color: #388e3c; font-size: 20px; font-weight: bold; padding: 20px; text-align: center; border-radius: 8px;'>This tweet is <b>Viral</b>!</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background-color: #ffccbc; color: #d32f2f; font-size: 20px; font-weight: bold; padding: 20px; text-align: center; border-radius: 8px;'>This tweet is <b>Non-Viral</b>!</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a tweet to predict its virality.")

# Footer
st.markdown("""
    <div style='text-align: center; color: #777; font-size: 14px; margin-top: 30px;'>
        🚀 <b>Tweet Virality Predictor</b> by <i>Safiullah</i><br>
        Powered by Streamlit &amp; Logistic Regression Model
    </div>
""", unsafe_allow_html=True)
