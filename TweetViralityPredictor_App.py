import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib  # For loading the saved model
import re

# Function to count hashtags
def count_hashtags(tweet):
    hashtags = re.findall(r'\#\w+', tweet)  # Find all hashtags
    return len(hashtags)

# Load pre-trained model and TF-IDF vectorizer
model = joblib.load('logistic_regression_model.pkl')  # Save the model using joblib
tfidf = joblib.load('tfidf_vectorizer.pkl')  # Save the TF-IDF vectorizer using joblib

# Define Streamlit app interface
st.title("Tweet Virality Prediction")

st.write("""
# Enter a tweet and predict if it will be viral or not.
""")

# User input for tweet content
tweet = st.text_area("Enter Tweet", "Type your tweet here...")

if tweet:
    # Extract number of hashtags from the tweet
    num_hashtags = count_hashtags(tweet)

    # Transform tweet using the TF-IDF vectorizer
    tweet_tfidf = tfidf.transform([tweet]).toarray()

    # Combine the TF-IDF vector and the number of hashtags as input for prediction
    X_input = np.hstack((tweet_tfidf, np.array([[num_hashtags]])))

    # Make a prediction
    prediction = model.predict(X_input)

    # Display the result
    if prediction[0] == 1:
        st.write("This tweet is **Viral**!")
    else:
        st.write("This tweet is **Non-Viral**!")
