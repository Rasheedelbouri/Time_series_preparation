#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:41:54 2020

@author: kebl4170
"""

from data_loading import generateProfiles, plot


def cleanData(dataframe):
    cols = ['HR', 'RR','SBP','SPO2','TEMP','avpu','age']
    for feat in cols:
        dataframe[feat] = (dataframe[feat] - min(dataframe[feat])) / (max(dataframe[feat]) - min(dataframe[feat]))
    dataframe['sex'] = dataframe['sex'].map({'F': 1, 'M': 0})
    return(dataframe)

gp = generateProfiles(read=True,normalised=True, edition=1)
survivors, survivor_ids, deaths, death_ids = gp.getDataSplit()





