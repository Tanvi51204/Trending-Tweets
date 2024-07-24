# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier  # Or whatever model class you're using
# # Optionally import numpy if needed
# import numpy as np
# import pickle

# with open('naive_bayes_model-4.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# # Load the TF-IDF Vectorizer
# with open('tfidf_vectorizer-4.pkl', 'rb') as vectorizer_file:
#     vectorizer = pickle.load(vectorizer_file)


# # Function to make predictions
# def predict_trend(retweet_count, user_followers_count, tweet_text):
#     # Transform the text input to a TF-IDF vector
#     text_features = vectorizer.transform([tweet_text])
    
#     # Combine the features into a single array
#     features = pd.DataFrame({
#         'retweet_count': [retweet_count],
#         'user_followers_count': [user_followers_count]
#     })
    
#     # Combine numerical features with text features
#     # Assuming you concatenate them horizontally
#     full_features = pd.concat([features, pd.DataFrame(text_features.toarray())], axis=1)
#     full_features.columns = full_features.columns.astype(str)
    
#     # Predict using the model
#     prediction = model.predict(full_features)
#     return prediction[0]

# # Streamlit UI
# st.title("Trending Tweet Predictor")

# st.write("Enter the tweet details below to predict if the hashtag is trending.")

# retweet_count = st.number_input("Retweet Count", min_value=0, step=1)
# user_followers_count = st.number_input("User Followers Count", min_value=0, step=1)
# tweet_text = st.text_area("Tweet Text")

# if st.button("Predict"):
#     prediction = predict_trend(retweet_count, user_followers_count, tweet_text)
#     if prediction == 1:
#         st.success("The tweet is trending!")
#     else:
#         st.info("The tweet is not trending.")

import streamlit as st
import pandas as pd
import pickle

# Load the models and vectorizer
with open('naive_bayes_model-5.pkl', 'rb') as nb_file:
    nb_classifier, feature_names_in_nb = pickle.load(nb_file)

with open('random_forest_model-5.pkl', 'rb') as rf_file:
    rf_classifier, feature_names_in_rf = pickle.load(rf_file)

with open('tfidf_vectorizer-5.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to make predictions
def predict_trend(retweet_count, user_followers_count, tweet_text, model_type='rf'):
    # Transform the text input to a TF-IDF vector
    text_features = vectorizer.transform([tweet_text])
    
    # Create DataFrame for text features
    text_features_df = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Create DataFrame for additional features
    features_df = pd.DataFrame({
        'retweet_count': [retweet_count],
        'user_followers_count': [user_followers_count]
    })
    
    # Combine numerical features with text features
    full_features = pd.concat([text_features_df, features_df], axis=1)
    
    # Ensure columns are in the same order as during training
    if model_type == 'nb':
        full_features = full_features[feature_names_in_nb]
        # Predict using Naive Bayes model
        prediction = nb_classifier.predict(full_features)
    elif model_type == 'rf':
        full_features = full_features[feature_names_in_rf]
        # Predict using Random Forest model
        prediction = rf_classifier.predict(full_features)
    return prediction[0]

# Streamlit UI
st.title("Hashtag Trend Predictor")

st.write("Enter the tweet details below to predict if the hashtag is trending.")

retweet_count = st.number_input("Retweet Count", min_value=0, step=1)
user_followers_count = st.number_input("User Followers Count", min_value=0, step=1)
tweet_text = st.text_area("Tweet Text")
model_type = st.radio("Choose model", ('Naive Bayes', 'Random Forest'))

if st.button("Predict"):
    prediction = predict_trend(retweet_count, user_followers_count, tweet_text, model_type='nb' if model_type == 'Naive Bayes' else 'rf')
    if prediction == 1:
        st.success("The hashtag is trending!")
    else:
        st.info("The hashtag is not trending.")
