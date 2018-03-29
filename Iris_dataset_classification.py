#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 00:14:36 2018

@author: ayush
"""

import pandas
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

dataset = pandas.read_csv(url, names=names)

#Observe with dataset
print(dataset.shape)
print(dataset.head(20))

#Following command prints all the dataset at once:-
print(dataset.head)

#An important command:-
print(dataset.describe())
#Similar as previous case, this command without parenthesis prints all data at once

#Group by any class in the list 'names'
print(dataset.groupby('class').size())

#Visualising the dataset:-
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
#Note:- sharex and sharey force same scale on all x axes or all y axes resp.

dataset.hist()
plt.show()

#The above task can also be done using previous fxn as follows:-
dataset.plot(kind='hist', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#Scatter Matrix:-
scatter_matrix(dataset)
plt.show()


array = dataset.values
#This command groups the data row wise into 'array' whise data type is object.

X= array[:,0:4]
Y= array[:,4]
validation_size=0.2#amount of data required in validation set
seed=7#just a seed for random number generator
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size, random_state=seed)

scoring = 'accuracy'

models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('SVM',SVC()))

results=[]
names=[]

for name, model in models:
    kfold= model_selection.KFold(n_splits=10, random_state=seed)
    #kfold contains indices to the splitted dataset for cross validation
    #random_state passed here is useless as 'shuffle' by default is off
    #Check that its absence does not affect the result.
    cv_results=model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg= "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print (msg)
    
    
fig=plt.figure()
fig.suptitle('Algorithm comparision')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

knn= KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions= knn.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
