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


#move the prediction made with a laptop from a computer from another
#it is done with SERIALIZATION
#have all in one file and move the file to another computer

import json
a = {'this': 1, 'is':2, 'a': 3, 'test':4}  #maps keys to values
json_a = json.dumps(a)
json_a #now the dictionary is a string, it expects double qotes and {} and space between values and keys
#json has a very strict format, in fact to use json we are going to use the library to don't make mistake
with open('test.json','w') as fh:
    fh.write(json_a)              #create the file
#json serialization can be used with all kind of objects out of functions
b = {(1,2):'what?'}
b
json.dumps(b) #it gives an error
del b  #delet b variable

json_loads = json.loads(json_a)  #it gives the original dictionary
json_loads

c = {1:2,3:4}
json_c = json.dumps(c)
json_c

#in the new computer
import json
json.loads(json_c) #to have back the original dictionary in the new laptop

#if the value is a linear regression and we pass it thought json we loose the fact that is a regression but we keep the coefficients.
#json very good with dictionary, doesn't store the object type



import pickle
zoo = {'lion':2,
       'elephant':10,
       'zebra':25}
zoo
p_zoo = pickle.dumps(zoo)  #it stores only the name of the class,
pickle.loads(p_zoo)
#pickels is different form json: the output of pickels is not readable by human
#with pickle we can put inside also a file
#the first numbers of pickle refers to the class
#it could give back the info of the linear regression
#another advantaj is that the string we create with json there is a normal string, here there is a bit string so smaller pieces
#and it allows us to save some spaces and store the information in less ram
#pickle so is useful to move data from one place to another
#there are different version of pickle, from 1 to 6, the default is 1. it stores in different way the dumps function
#pickle store the object type


from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
X,y = datasets.load_iris(return_X_y= True)
clf.fit(X,y)

from joblib import dump, load
dump(clf, 'model.joblit')
#it saves a file with the name of the class and a reference of the class so it is useful for thigs that doesn't change like regression that are fixed and based on formulas
clf2 = load('model.joblit')
clf2  #load back the file created

import ast
d = ast.literal_eval(json_a)
d #it gives back the dictionary

