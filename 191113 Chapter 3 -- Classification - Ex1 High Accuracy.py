#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:05:14 2019

@author: ai
"""

#%% Set random seed 
import numpy as np
np.random.seed(42)

#%% CLASSIFICATION

#%% MNIST
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys() 

#%%
X, y = mnist["data"], mnist["target"]

#%% Splitting into training and test sets

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#%% KNN

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

y_pred = knn_clf.predict(X_test)
