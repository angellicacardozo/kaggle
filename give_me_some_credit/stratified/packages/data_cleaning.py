#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 01:51:59 2020

@author: angellica
"""
import pandas

def normalize(df, feature):
    min_pastdue = df[feature].min()
    max_pastdue = df[feature].max()
    
    # Converto to 0 1 interval
    df[feature] = df.apply(lambda x: (x[feature] - min_pastdue)/(max_pastdue - min_pastdue), axis=1)
    return df

def _flagandfill(df):
    features = ["DebtRatio", "NumberOfDependents", "MonthlyIncome", "RevolvingUtilizationOfUnsecuredLines"]
    for i in range(len(features)):
        na_label = "na_" + features[i]
    
        df[na_label] = df.apply(lambda x: 1 if pandas.isna(x[features[i]]) else 0, axis=1)
        df.fillna({features[i]: 0}, inplace=True)
        
    return df

def run(df, df_test):
    df = _flagandfill(df)
    df_test = _flagandfill(df_test)
    
    df = df[(df['NumberOfDependents'] <= 6)]
    df = df[(df['RevolvingUtilizationOfUnsecuredLines'] <= 1)]
    df = df[(df['age'] >= 20) & (df['age'] <= 95)]
    df = normalize(df, "MonthlyIncome")
    
    return df, df_test