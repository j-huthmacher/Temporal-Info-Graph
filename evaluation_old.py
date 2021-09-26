"""
    @source: https://github.com/fanyun-sun/InfoGraph/
    @author: j-huthmacher
"""
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing   
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from sklearn.decomposition import PCA

from model.solver import evaluate  

from tqdm import tqdm


def mlp_classify(x,y, search=False, ret_pred=False):
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    n_splits = min(min(10, x.shape[0]), min(np.unique(y, return_counts=True)[1]))
    n_splits = max(2, n_splits)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in tqdm(kf.split(x, y)):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'activation':['tanh', 'relu'], "learning_rate_init": [0.1, 0.01, 0.001, 0.0001]}
            classifier = GridSearchCV(MLPClassifier(), params, cv=n_splits, scoring='accuracy', verbose=0)
        else:
            classifier = MLPClassifier()
        classifier.fit(x_train, y_train)
        yhat = classifier.predict_proba(x_test)
        # accuracies.append(accuracy_score(y_test, yhat))
        # accuracies.append((top_k_accuracy_score(y_test, yhat, k = 1),
                        #    top_k_accuracy_score(y_test, yhat, k = 5)))
        accuracies.append(evaluate(np.argsort(yhat, axis=1), y_test))

    if ret_pred:
        return classifier.predict(x), np.mean(np.asarray(accuracies), axis=0)
    else:
        return np.mean(np.asarray(accuracies), axis=0)


def svc_classify(x, y, search, ret_pred=False):
    """ From InfoGraph
    """
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    n_splits = min(min(10, x.shape[0]), min(np.unique(y, return_counts=True)[1]))
    n_splits = max(2, n_splits)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            classifier = GridSearchCV(SVC(probability=True), params, cv=n_splits, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10, probability=True)
        classifier.fit(x_train, y_train)
        # yhat = classifier.predict(x_test)
        yhat = classifier.predict_proba(x_test)
        # accuracies.append(accuracy_score(y_test, yhat))

        accuracies.append(evaluate(np.argsort(yhat, axis=1), y_test))

    if ret_pred:
        return classifier.predict(x), np.mean(np.asarray(accuracies), axis=0)
    else:
        return np.mean(np.asarray(accuracies), axis=0)


def randomforest_classify(x, y, search, ret_pred=False):
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    n_splits = min(min(10, x.shape[0]), min(np.unique(y, return_counts=True)[1]))
    n_splits = max(2, n_splits)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in tqdm(kf.split(x, y)):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=min(5, n_splits), scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        yhat = classifier.predict(x_test)
        accuracies.append(accuracy_score(y_test, yhat))
        # accuracies.append((top_k_accuracy_score(y_test, yhat, k = 1),
                        #    top_k_accuracy_score(y_test, yhat, k = 5)))

    if ret_pred:
        return classifier.predict(x), np.mean(np.asarray(accuracies), axis=0)
    else:
        return np.mean(np.asarray(accuracies), axis=0)


def linearsvc_classify(x, y, search, ret_pred = False):
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    n_splits = min(min(10, x.shape[0]), y.shape[1])
    n_splits = max(2, n_splits)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        classifier.fit(x_train, y_train)
        yhat = classifier.predict(x_test)
        accuracies.append(accuracy_score(y_test, yhat))
        # accuracies.append((top_k_accuracy_score(y_test, yhat, k = 1),
                        #    top_k_accuracy_score(y_test, yhat, k = 5)))
    if ret_pred:
        return classifier.predict(x), np.mean(np.asarray(accuracies), axis=0)
    else:
        return np.mean(np.asarray(accuracies), axis=0)


def evaluate_embedding(embeddings, labels, search=True):
    """
    """
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    # print(x.shape, y.shape)

    logreg_accuracies = [logistic_classify(x, y) for _ in range(1)]
    # print(logreg_accuracies)
    print('LogReg', np.mean(logreg_accuracies))

    svc_accuracies = [svc_classify(x,y, search) for _ in range(1)]
    # print(svc_accuracies)
    print('svc', np.mean(svc_accuracies))

    linearsvc_accuracies = [linearsvc_classify(x, y, search) for _ in range(1)]
    # print(linearsvc_accuracies)
    print('LinearSvc', np.mean(linearsvc_accuracies))

    randomforest_accuracies = [randomforest_classify(x, y, search) for _ in range(1)]
    # print(randomforest_accuracies)
    print('randomforest', np.mean(randomforest_accuracies))

    return np.mean(logreg_accuracies), np.mean(svc_accuracies), np.mean(linearsvc_accuracies), np.mean(randomforest_accuracies)