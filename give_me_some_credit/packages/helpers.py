#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 16:33:46 2020

@author: angellica
"""
import pandas

def save_as_kaggle_result(path, model, X_test):
    y_pred = model.predict_proba(X_test)
    prediction = pandas.DataFrame({"Id": df_test.index, "Probability": y_pred[:, 1]})
    prediction.to_csv(path, index=False)  