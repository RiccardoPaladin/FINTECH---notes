'''
MID-TERM EXAM
FA 595
'''
#https://docs.streamlit.io/library/get-started/create-an-app
#AWS Amplify hosting --> Amplify Hosting is a fully managed hosting service for web apps. Connect your repository to build, deploy, and host your web app.



#pip install streamlit

import pandas as pd
import numpy as np
import requests
import streamlit as st

st.title('NLP services')
st.text('In this web app you can use six different services on NLP')

st.text('FIRST SERVICE')
st.text('Insert a string and obtain which part of the speech is')
#Tokenize the sentence and return the pos_tag
text_input = st.text_input('Enter string to analyze','')

import nltk
nltk.download('punkt')
tokens = nltk.word_tokenize(text_input)
tag = nltk.pos_tag(tokens)
st.markdown(
    f"""
    {tag}
    """
)

st.text('SECOND SERVICE')
st.text('Insert a string and obtain the sentiment analysis')
text_input1 = st.text_input('Enter string to evaluate','')
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores(text_input1)
st.markdown(
    f"""
    {sentiment}
    """
)



#streamlit run MID_TERM_FA_595    on the terminal
#Local URL: http://localhost:8501
#Network URL: http://192.168.12.117:8501
