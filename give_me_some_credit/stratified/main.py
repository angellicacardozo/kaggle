#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 01:33:22 2020

@author: angellica
"""

import os
import pandas
from sklearn.model_selection import train_test_split

import data_cleaning
import feature_engineering
import tunning

#*************************************************************************
#   CONFIGURE PATHS
#*************************************************************************
root_path = os.path.abspath("..")

trainning_data_path = root_path + "/data/cs-training.csv"
testing_data_path = root_path + "/data/cs-test.csv"

#*************************************************************************
#   LOAD DATASET
#*************************************************************************
df = pandas.read_csv(trainning_data_path, index_col=0)
df_test = pandas.read_csv(testing_data_path, index_col=0)

#*************************************************************************
#   PERFORM DATA CLEANING
#*************************************************************************
data_cleaning.run(df, df_test)

#*************************************************************************
#   PERFORM FEATURE ENGINEERING
#*************************************************************************
feature_engineering.run(df, df_test)

#*************************************************************************
#   SPLIT TRAIN AND TEST SAMPLES
#*************************************************************************
features = ['age', 
            'NumberOfTime30-59DaysPastDueNotWorse', 
            'MonthlyIncome', 
            'NumberOfDependents', 
            'na_NumberOfDependents', 
            'na_MonthlyIncome', 
            'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'pastdue_score']

input_x = df[features]
input_y = df['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(input_x, input_y, test_size=0.5, random_state = 0, stratify = input_y, shuffle = True)

#*************************************************************************
#   PERFORM TUNNING PARAMETERS
#*************************************************************************
tunned_rf = tunning.run(X_train, y_train)