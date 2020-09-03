#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:53:27 2020

@author: kebl4170
"""

import pandas as pd
import os
import pickle
from timestamps_included import separatePatients
from visualise_records import visualiser
import sys

def loadEpisodes(path):
    episodes = pd.read_csv(os.path.join(path, 'episodes.csv'))
    return(episodes)


def load(path):
    events = pd.read_csv(os.path.join(path, 'all_event_obs.csv'))
    return(events)
    
def cleanData(dataframe):
    cols = ['HR', 'RR','SBP','SPO2','TEMP','avpu','age']
    for feat in cols:
        dataframe[feat] = (dataframe[feat] - min(dataframe[feat])) / (max(dataframe[feat]) - min(dataframe[feat]))
    dataframe['sex'] = dataframe['sex'].map({'F': 1, 'M': 0})
    
    return(dataframe)

def selectFeatures(dataframe, cols):
    dataframe = dataframe[cols]
    return(dataframe)
    
    
def sort(dataframe):
    sp = separatePatients()
    records = sp.separate(dataframe)
    return(records)
    
def plot(records, comparison, vitals='HR', patient_id=2474):
    v = visualiser(records, comparison, patient_id)
    v.plotVitals(vitals)    

class generateProfiles():
    
    def __init__(self, read=True, normalised=True, edition=2):
        
        if  isinstance(read, bool) and edition in (1,2):
            self.read = read
            self.edition = edition
            self.path = '../../../netshares/ibme/Projects_1/orchid/processing/SK/data_v'
            self.normalised = normalised
            self.norm = str("")

        else:
            sys.exit("read must be a boolean and edition must be either 1 or 2")
                
    def getDataSplit(self):
    

        path = self.path + str(self.edition)
    
        if self.read == True:
            if self.normalised == True:
                self.norm = str("_normalised")
            if self.edition == 1:
                with open("records1" + self.norm + ".pkl", "rb") as f:
                    records = pickle.load(f)
            elif self.edition == 2:
                with open("records2" + self.norm + ".pkl", "rb") as f:
                    records = pickle.load(f)
            ids = list(records.keys())
            
        else:
        
            all_cols = ['HR', 'RR','SBP','SPO2','TEMP','avpu','age', 'sex', 'seconds']
            events = load(path)
            events = cleanData(events)
            records = sort(events)
            ids = list(records.keys())
            for k in ids:
                records[k] = selectFeatures(records[k], all_cols)
                
        episodes = loadEpisodes(path)
        survivors = episodes[episodes.LinkedDeathdate.isnull()]
        deaths = episodes[~episodes.LinkedDeathdate.isnull()]
        
        survivors = survivors[survivors.ClusterID.isin(list(records.keys()))]
        deaths = deaths[deaths.ClusterID.isin(list(records.keys()))]
        
        survs = dict((k, records[k]) for k in survivors.ClusterID)
        dead = dict((k, records[k]) for k in deaths.ClusterID)
        
        surv_ids = pd.DataFrame(list(survs.keys()))
        dead_ids = pd.DataFrame(list(dead.keys()))
        
        return(survs, surv_ids, dead, dead_ids)
        
