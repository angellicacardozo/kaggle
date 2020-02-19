#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 01:51:59 2020

@author: angellica
"""
import pandas

def _flagandfill(df):
    df['na_NumberOfDependents'] = df.apply(lambda x: 1 if pandas.isna(x['NumberOfDependents']) else 0, axis=1)
    df['na_MonthlyIncome'] = df.apply(lambda x: 1 if pandas.isna(x['MonthlyIncome']) else 0, axis=1)
    df.fillna({'NumberOfDependents': 0}, inplace=True)
    df.fillna({'MonthlyIncome': 0}, inplace=True)    
    return df

def run(df, df_test):
    # 1 Flag and fill all nans (only for training)
    df = _flagandfill(df)
    df_test = _flagandfill(df_test)
    
    # 2 Remove outliers by number of dependents (only for training)
    df = df[(df['NumberOfDependents'] <= 6)]
    df = df[(df['age'] >= 20) & (df['age'] <= 95)]