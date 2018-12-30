# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 18:09:03 2018

@author: Deepak
"""
#loading all modules
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Loading the dataset
digits=datasets.load_digits()


#preparing the classifier
classifierSVM=svm.SVC(gamma=0.001,C=1000)
classifierRandomForest=RandomForestClassifier()
classifierKNeighbors=KNeighborsClassifier()
classifierLogisticRegression=LogisticRegression()
classifierMultinomialNB=MultinomialNB()
classifierDecisionTree=DecisionTreeClassifier()


#getting the training and test set
train_X,test_X,train_Y,test_Y=train_test_split(digits.data,digits.target,test_size=.25)


#training the classifiers
classifierSVM.fit(train_X,train_Y)
classifierRandomForest.fit(train_X,train_Y)
classifierKNeighbors.fit(train_X,train_Y)
classifierLogisticRegression.fit(train_X,train_Y)
classifierMultinomialNB.fit(train_X,train_Y)
classifierDecisionTree.fit(train_X,train_Y)


predictions=classifierSVM.predict(test_X)
print("SVM",accuracy_score(test_Y,predictions))

predictions=classifierRandomForest.predict(test_X)
print("Random Forest",accuracy_score(test_Y,predictions))

predictions=classifierKNeighbors.predict(test_X)
print("K Nearest Neighbor",accuracy_score(test_Y,predictions))

predictions=classifierLogisticRegression.predict(test_X)
print("Logistic Regression",accuracy_score(test_Y,predictions))

predictions=classifierMultinomialNB.predict(test_X)
print("Multinomial Naive Baes",accuracy_score(test_Y,predictions))

predictions=classifierDecisionTree.predict(test_X)
print("Classifier Decision Tree",accuracy_score(test_Y,predictions))
