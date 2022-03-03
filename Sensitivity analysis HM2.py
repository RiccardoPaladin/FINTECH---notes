"""
Analyze best and worst companies based on their purposes
"""
import pandas as pd
import numpy as np
from textblob import TextBlob

data = pd.read_csv(r'/Users/riccardopaladin/Desktop/FINTECH/webscrape_companies.csv')
data

import nltk
nltk.download()
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analysis = []
for purpose in data['Purposes']:
    analysis = []
    sia = SentimentIntensityAnalyzer()
    analysi = sia.polarity_scores(purpose)
    analysis.append(analysi)



print(p)
sia = SentimentIntensityAnalizer
sia.polarity_scores("this is a string.")

