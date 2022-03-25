"""
Analyze best and worst companies based on their purposes
"""
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

data = pd.read_csv(r'/Users/riccardopaladin/Desktop/FINTECH/webscrape_companies.csv')
data = pd.DataFrame(data.iloc[0:,1:])


def best_worst_company(data):
    '''
    Function that has as input a dataset with name and purposes of 50 companies
    :return: Data Frame with companies ordered by sentiment analysis
    '''
    best_worst_company = pd.DataFrame()
    purposes = []
    for purpose in data['Purposes']:
        purposes.append(purpose)

    analysis = []
    for i in purposes:
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(i)
        analysis.append(sentiment)

    sentiments = pd.DataFrame(analysis)
    data_tot = pd.concat([data, sentiments], axis=1)
    best_worst_company = data_tot.sort_values(["compound", "Names"], ascending=False)
    return best_worst_company

best_worst_company = best_worst_company(data)
print(best_worst_company.info())

#Subsets of best and worst ten companies
best_10 = best_worst_company.iloc[0:10,:1]
worst_10 = best_worst_company.iloc[40:,:1]
print('The best 10 companies are\n', best_10)
print('The worst 10 companies are\n', worst_10)
