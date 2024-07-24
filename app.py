import streamlit as st
import pandas as pd
import pickle


with open('naive_bayes_model-5.pkl', 'rb') as nb_file:
    nb_classifier, feature_names_in_nb = pickle.load(nb_file)

with open('random_forest_model-5.pkl', 'rb') as rf_file:
    rf_classifier, feature_names_in_rf = pickle.load(rf_file)

with open('tfidf_vectorizer-5.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_trend(retweet_count, user_followers_count, tweet_text, model_type='rf'):
    text_features = vectorizer.transform([tweet_text])
    text_features_df = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())
 
    features_df = pd.DataFrame({
        'retweet_count': [retweet_count],
        'user_followers_count': [user_followers_count]
    })

    full_features = pd.concat([text_features_df, features_df], axis=1)
    if model_type == 'nb':
        full_features = full_features[feature_names_in_nb]
        prediction = nb_classifier.predict(full_features)
    elif model_type == 'rf':
        full_features = full_features[feature_names_in_rf]
        prediction = rf_classifier.predict(full_features)
    return prediction[0]

# UI
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
