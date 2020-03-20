#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 01:33:22 2020

@author: angellica
"""

import sys
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import data_cleaning
import feature_engineering
import ploting
import tunning
import helpers

#*************************************************************************
#   CONFIGURE PATHS
#*************************************************************************
root_path = "<your_path_here>/kaggle/give_me_some_credit"

common_packages_path = root_path + "/stratified/packages"
module_packages_path = root_path + "/packages"
files_path = root_path + "/results"

sys.path.append(common_packages_path)
sys.path.append(module_packages_path)

#*************************************************************************
#   LOAD DATASET
#*************************************************************************
trainning_data_path = root_path + "/data/cs-training.csv"
testing_data_path = root_path + "/data/cs-test.csv"

df = pandas.read_csv(trainning_data_path, index_col=0)
df_test = pandas.read_csv(testing_data_path, index_col=0)

#*************************************************************************
#   FEATURE TO USE
#*************************************************************************
features = ['age', 
            'DebtRatio',
            'NumberOfTime30-59DaysPastDueNotWorse', 
            'MonthlyIncome', 
            'NumberOfDependents', 
            'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'RevolvingUtilizationOfUnsecuredLines']

#*************************************************************************
#   EXPLORATORY ANALYSIS
#*************************************************************************
input_x = df[features]
input_y = df['SeriousDlqin2yrs']

ploting.class_balance_pie(input_y)

#*************************************************************************
#   PERFORM DATA CLEANING
#*************************************************************************
df, df_test = data_cleaning.run(df, df_test)

#*************************************************************************
#   PERFORM FEATURE ENGINEERING
#*************************************************************************
df, df_test = feature_engineering.run(df, df_test)

#*************************************************************************
#   SPLIT TRAIN AND TEST SAMPLES
#*************************************************************************
input_x = df[features]
input_y = df['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(input_x, input_y, test_size=0.3333, random_state = 0, stratify = input_y, shuffle = True)
ploting.class_balance_pie(y_train)

#*************************************************************************
#   PERFORM TUNNING PARAMETERS
#*************************************************************************
model = RandomForestClassifier()
tunned_model = tunning.run(model, X_train, y_train)

#*************************************************************************
#   PERFORM TRAINNING
#*************************************************************************
tunned_model.fit(X_train, y_train)
y_pred = tunned_model.predict(X_test)
y_score = tunned_model.fit(X_train, y_train).predict_proba(X_test)

#*************************************************************************
#   PLOT RESULTS
#*************************************************************************
ploting.roc_curve_score(y_test, y_score[:,1])
ploting.confusion_matrix(y_test, y_pred, [0, 1], normalize=False, title='CM FOR OPTIMIZED CLASSIFIER')

#*************************************************************************
#   SAVE FILES
#*************************************************************************
helpers.save_as_kaggle_result(files_path + "/kaggle_result.csv", model, X_test)
