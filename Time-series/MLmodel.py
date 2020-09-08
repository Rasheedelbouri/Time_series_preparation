#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:49:45 2020

@author: kebl4170
"""

from training_setup import generateTrainTestSplits
from model.network_building import buildNetwork
import numpy as np
import pandas as pd


def expandDims(arrays):
    
    for i in range(0,len(arrays)):
        arrays[i] = np.expand_dims(arrays[i], axis=0)
        
    return(arrays)

def dict2list(dictionary):
    
    dictlist = []
    for value in dictionary.values():
        temp = np.array(value)
        dictlist.append(temp)
    return(dictlist)

def trainingData():
    g = generateTrainTestSplits(split=0.6, edition=2)
    strx, stex, dtrx, dtex = g.combine(change_points=True)
    stry, stey, dtry, dtey = np.zeros(len(strx)), np.zeros(len(stex)), np.ones(len(dtrx)), np.ones(len(dtex))

    train_x = {**strx, **dtrx}
    train_x = {i: v for i, v in enumerate(train_x.values())}
    train_y = pd.DataFrame(np.concatenate((stry, dtry),axis=0))
    
    test_x = {**stex, **dtex}
    test_x = {i: v for i, v in enumerate(test_x.values())}
    test_y = pd.DataFrame(np.concatenate((stey, dtey),axis=0))

    
    return(train_x, test_x, train_y, test_y)
    
    

if __name__ == "__main__":
    
    train_x, test_x, train_y, test_y = trainingData()
    
    train_x, test_x = dict2list(train_x), dict2list(test_x)
    
    bn = buildNetwork(source='M')
    model = bn.build(train_x, np.zeros(2))
    model = bn.compiler(model,q_net=False,actor=False)
    model.summary()
    
    train_x = expandDims(train_x)
    test_x = expandDims(test_x)
    
    
    
    loss = []
    acc = []
    
    for epoch in range(0,5):
        train_y = train_y.sample(frac=1)
        indices = list(train_y.index)
        for i in indices:
            x,y = model.train_on_batch(train_x[i], [train_y[0][i]])
            loss.append(x)
            acc.append(y)
