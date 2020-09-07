# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:42:23 2020

@author: rashe
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


def separatePatients(dataframe):

    time_feats = ['admittime', 'dischtime', 'eventtime', 'charttime']
    for tf in time_feats:
        dataframe[tf] = pd.to_datetime(dataframe[tf])
    
    unique_patients = pd.DataFrame(dataframe.hadm_id.unique())
    unique_patients = unique_patients.reset_index(drop=True)
    
    patient_records = dict()
    for p in unique_patients[0]:
        patient_records[p] = dataframe[dataframe.hadm_id == p]
        patient_records[p] = patient_records[p].sort_values('charttime')
        patient_records[p] = patient_records[p].reset_index(drop=True)
        patient_records[p]['timedelta'] = np.zeros(len(patient_records[p]))
        patient_records[p]['seconds'] = np.zeros(len(patient_records[p]))
        eventtimes = [0]
    
        for i in range(len(patient_records[p])):
            if i == 0:
                patient_records[p]['timedelta'][i:i+1] = patient_records[p]['charttime'][i] - patient_records[p]['charttime'][i]
            else:
                patient_records[p]['timedelta'][i:i+1] = patient_records[p]['charttime'][i] - patient_records[p]['charttime'][i-1]
                patient_records[p]['seconds'][i:i+1] = patient_records[p]['timedelta'][i:i+1][i].seconds
                if ((patient_records[p]['seconds'][i:i+1][i] == 0) and (sum(eventtimes) == 0)):
                    continue
                elif (patient_records[p]['seconds'][i:i+1][i] != 0):
                    eventtimes.append(patient_records[p]['seconds'][i:i+1][i])
                    patient_records[p]['seconds'][i:i+1] = sum(eventtimes)
                elif (patient_records[p]['seconds'][i:i+1][i] == 0) & (sum(eventtimes) != 0):
                    patient_records[p]['seconds'][i:i+1] = sum(eventtimes)
    return(patient_records)
        
