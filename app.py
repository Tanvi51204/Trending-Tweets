import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  # Or whatever model class you're using
# Optionally import numpy if needed
import numpy as np
import pickle

with open('naive_bayes_model-4.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF Vectorizer
with open('tfidf_vectorizer-4.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Function to make predictions
def predict_trend(retweet_count, user_followers_count, tweet_text):
    # Transform the text input to a TF-IDF vector
    text_features = vectorizer.transform([tweet_text])
    
    # Combine the features into a single array
    features = pd.DataFrame({
        'retweet_count': [retweet_count],
        'user_followers_count': [user_followers_count]
    })
    
    # Combine numerical features with text features
    # Assuming you concatenate them horizontally
    full_features = pd.concat([features, pd.DataFrame(text_features.toarray())], axis=1)
    
    # Predict using the model
    prediction = model.predict(full_features)
    return prediction[0]

# Streamlit UI
st.title("Trending Tweet Predictor")

st.write("Enter the tweet details below to predict if the hashtag is trending.")

retweet_count = st.number_input("Retweet Count", min_value=0, step=1)
user_followers_count = st.number_input("User Followers Count", min_value=0, step=1)
tweet_text = st.text_area("Tweet Text")

if st.button("Predict"):
    prediction = predict_trend(retweet_count, user_followers_count, tweet_text)
    if prediction == 1:
        st.success("The tweet is trending!")
    else:
        st.info("The tweet is not trending.")
