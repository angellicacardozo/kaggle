#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:12:50 2020

@author: angellica
"""
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

def run(rf, X_train, y_train):
    n_estimators    = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features    = ['auto', 'sqrt']
    max_depth       = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split   = [2, 5, 10]
    min_samples_leaf    = [1, 2, 4]
    bootstrap           = [True, False]
    scoring_list = {'AUC': 'roc_auc', 'Precision':  'precision_weighted', 'Accuracy': 'accuracy', 'F1_W': 'f1_weighted'}
    
    random_grid = {'n_estimators': n_estimators,
                   'criterion': ['gini', 'entropy'],
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'class_weight': [{0:1,1:2}, "balanced_subsample"]}
    
    tunned_rf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=scoring_list, refit='AUC')
    tunned_rf.fit(X_train, y_train)
    
    return tunned_rf