Trending Tweet Predictor

This project aims to predict whether a tweet is trending or not. The application is built using machine learning models, specifically Naive Bayes and Random Forest classifiers, and is deployed as a web application using Streamlit.


Project Overview

The Treding Tweet Predictor analyzes tweet data to predict if it is trending. This prediction is based on features such as the number of retweets, user follower count, and tweet content. The project leverages machine learning techniques to classify tweets as trending or not trending.


Application

The project includes a web application built with Streamlit, allowing users to input tweet details and predict if the Tweet is trending.

Features
User Input: Users can input retweet count, user followers count, and tweet text.
Prediction: The app predicts and displays whether the tweet is trending or not.
Interactive UI: Simple and interactive user interface built with Streamlit.
Usage

Prerequisites
Python 3.7 or higher
Required Python packages (install via requirements.txt)

Installation
Clone the repository:

bash

`git clone https://github.com/Tanvi51204/Trending-Tweets.git`
`cd Trending-Tweets`

Create a virtual environment and activate it:

bash

`python -m venv env`
`source env/bin/activate`  

 On Windows use 
`env\Scripts\activate`


Install the dependencies:

bash

`pip install -r requirements.txt`

Launch the Streamlit application:

bash

`streamlit run app.py`

Results

Naive Bayes Classifier: Achieved reasonable accuracy with simpler computation.
Random Forest Classifier: Struggled with predictions, likely due to model configuration or feature handling.
The Naive Bayes model successfully identified trending hashtags in test examples, indicating its robustness in text-based classification.
