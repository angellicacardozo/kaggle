#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 02:07:34 2020

@author: angellica
"""

def engPastDueScore(df):
    df['pastdue_score'] = df.apply(lambda x: x['NumberOfTime30-59DaysPastDueNotWorse'] + x['NumberOfTime60-89DaysPastDueNotWorse'], axis=1)
    min_pastdue = df['pastdue_score'].min()
    max_pastdue = df['pastdue_score'].max()
    
    # Converto to 0 1 interval
    df['pastdue_score'] = df.apply(lambda x: (x['pastdue_score'] - min_pastdue)/(max_pastdue - min_pastdue), axis=1)
    
    return df

def run(df, df_test):
    df = engPastDueScore(df)
    df_test = engPastDueScore(df_test)