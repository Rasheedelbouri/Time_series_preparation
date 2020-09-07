#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:41:54 2020

@author: kebl4170
"""

from data_formatting.data_loading import generateProfiles
from sklearn.model_selection import train_test_split
import pandas as pd
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
        
    def combine(self):
        gp = generateProfiles(read=True,normalised=True, edition=2)
        survivors, _, deaths, _ = gp.getDataSplit()
        
        surv_train_x, surv_test_x = self.trainTestSplit(survivors)
        dead_train_x, dead_test_x = self.trainTestSplit(deaths)
        
        return(surv_train_x, surv_test_x, dead_train_x, dead_test_x)
        
if __name__ == "__main__":
    
    g = generateTrainTestSplits(0.6, 2)
    s_tr_x, s_te_x, d_tr_x, d_te_x = g.combine()