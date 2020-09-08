#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:41:54 2020

@author: kebl4170
"""

from data_formatting.data_loading import generateProfiles
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random


raw_path = '../../../netshares/ibme/Projects_1/orchid/raw/PreVent'
            
class generateTrainTestSplits():
    
    def __init__(self, split, edition):
        self.split = split

    def trainTestSplit(self, dictionary):
        
        train_x = dict(random.sample(dictionary.items(), int(self.split*len(dictionary))))
        
        exclusion = list(train_x.keys())
        for key in exclusion:
             dictionary.pop(key, None)
        
        test_x = dictionary
        
        return(train_x, test_x)
        
    @staticmethod
    def findChangePoints(dictionary):
        
        keys = list(dictionary.keys())
        for key in keys:
            dictionary[key] = dictionary[key].drop_duplicates().reset_index(drop=True)
        
        return(dictionary)
        
    
    @staticmethod
    def imputeValues(dictionary):
        cols = ['HR', 'RR', 'SBP', 'SPO2', 'TEMP', 'avpu']
        keys = list(dictionary.keys())
        
        for key in keys:
            for col in cols:
                if dictionary[key][col].isnull().any() == False:
                    continue
                else:
                    nullkeys = list(dictionary[key][col][dictionary[key][col].isnull()].index)
                    for n in nullkeys:
                        if n <= 1:
                            dictionary[key][col][n:n+1] = np.mean(dictionary[key][col])
                        else:
                            dictionary[key][col][n:n+1] = np.mean(dictionary[key][col][n-2:n])
        
        return(dictionary)
        
    
    def processData(self, dictionary):
        dictionary = self.findChangePoints(dictionary)
        dictionary = self.imputeValues(dictionary)
        
        return(dictionary)
                    
                    
        
    def combine(self, change_points=True):
        gp = generateProfiles(read=True, normalised=True, edition=2)
        survivors, _, deaths, _ = gp.getDataSplit()
        
        surv_train_x, surv_test_x = self.trainTestSplit(survivors)
        dead_train_x, dead_test_x = self.trainTestSplit(deaths)
        
        if change_points == True:
            surv_train_x = self.processData(surv_train_x)
            surv_test_x = self.processData(surv_test_x)
            dead_train_x = self.processData(dead_train_x)
            dead_test_x = self.processData(dead_test_x)

        
        return(surv_train_x, surv_test_x, dead_train_x, dead_test_x)

