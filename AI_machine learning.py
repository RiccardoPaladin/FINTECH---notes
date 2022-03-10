import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
#THE DATA COLLECTED HAS TO BE ORGANIZED IN A 2 D ARRAY SUCH AS [[],[],[]]
n = 100
x = np.arange(n)
rs = check_random_state(0)  #set the seed to have the same random number

y = rs.randint(-50,50, size= (n,)) + 50.0 *np.log1p(np.array(n))
#random number from -50 to 50 and choose 100 of then multiply 50 by the log plus 1
y
lr = LinearRegression()   #train the data randomly produced
lr.fit(x[:,np.newaxis],y)  #from [0,1,2] to [[0],[1],[2]]
lr.predict([[4.5]])
lr.coef_ #slope of the line
lr.intercept_ #intercept, only one value
lr.predict([[0]])  #it gives the intercept
lr.predict([[0],[2.5],[4.8],[6]])

#K-MEANS CLUSTERING  --> classification
#step by step the classes are moved in order to divided them from each other, the cluster is the alement that individualize each class
#UNSUPERVIDE MACHINE LEARNING : we don't know the y we have x1 and x2

from sklearn.cluster import Kmeans
X = np.array([[1,2],[1,4],[10,2],[10,4]])
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X)  #from fit to predict
kmeans.predict([[3,4],[1000,32222],[-122,-343]])  #we have 2 groups 0 and 1
kmeans.cluster_centers_  #see where they are near to
kmeans.inertia_  #when we increase the number of clusters the inertia goes down


