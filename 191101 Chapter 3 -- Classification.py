#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:01:46 2019

@author: ai
"""
#%% CLASSIFICATION

#%% MNIST

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys() 

#%%
X, y = mnist["data"], mnist["target"]
X.shape
y.shape

#%%
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

#%%
y[0]

import numpy as np
y = y.astype(np.uint8)

#%% Splitting into training and test sets

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


#%% Training a Binary Classfier

#%% Creating target vectors for the classification task

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#%% Creating Stochastic Gradient Descent classfier & training it

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

#%% Detect images of number 5

sgd_clf.predict([some_digit])


#%% Performance Measures

#%% Implementing Cross-Validation

'''Homemade Cross-Validation'''

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
    
#%% Evaluation of SGD Classifier model with K-fold cross-validation

from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

#%% Dumb classifier that classifies everything as 'not-5'

from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

#%% Confusion Matrix

#%% Predict classes on the training set with cross-validation

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

#%% Create confusion matrix

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

#%% An example of a perfect classifier

y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)

#%% Precision and Recall (Selectivity and Sensitivity)

from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

#%% Compute F1 score (harmonic mean)

from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)

#%% Applying decision function

y_scores = sgd_clf.decision_function([some_digit])
y_scores

threshold = 0
y_some_digit_pred = (y_scores > threshold)

threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred