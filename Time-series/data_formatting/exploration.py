# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:08:51 2020

@author: rashe
"""


import os
import pandas as pd
import matplotlib.pyplot as plt

def plotGraphs(patient_records, feature):

    if feature in ['heart rate', 'hr', 'HR', 'heart_rate']:
        for p in unique_patients[0]:    # heart rate
            plt.plot(patient_records[p]["Vital_Signs HR"])
        plt.xlabel("time step")
        plt.ylabel(str(feature))
        plt.show()
    
    elif feature in ['rr', 'RR', 'resp_rate', 'respiration_rate']:
        for p in unique_patients[0]:    # respiration rate
            plt.plot(patient_records[p]["Vital_Signs RR"])
        plt.xlabel("time step")
        plt.ylabel(str(feature))
        plt.show()
    
    elif feature in ['SBP', 'sbp', 'bp', 'blood_pressure']:
        for p in unique_patients[0]: # systolic blood pressure
            plt.plot(patient_records[p]["Vital_Signs SBP"])
        plt.xlabel("time step")
        plt.ylabel(str(feature))
        plt.show()
        
    elif feature in ['temp', 'TEMP', 'temperature']:
        for p in unique_patients[0]: # temperature
            plt.plot(patient_records[p]["Vital_Signs TEMP"])
        plt.xlabel("time step")
        plt.ylabel(str(feature))
        plt.show()

path = "../event_6-12-24_allfeatures/event_6-12-24_allfeatures"

train = pd.read_csv(os.path.join(path, "train_dataevents_N24_W24.csv"), sep=',')

features = list(train)

unique_patients = pd.DataFrame(train.hadm_id.unique())
unique_patients = unique_patients.sort_values(0).reset_index(drop=True)

patient_records = dict()

for p in unique_patients[0]:
    patient_records[p] = train[train.hadm_id == p]
    patient_records[p] = patient_records[p].reset_index(drop=True)

plotGraphs(patient_records, 'hr')
