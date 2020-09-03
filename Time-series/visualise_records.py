#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 12:02:44 2020

@author: kebl4170
"""

import matplotlib.pyplot as plt
import sys



class visualiser():
    
    def __init__(self, records, comparison, patient_id):
        
        self.records = records
        self.comparison = comparison
        self.patient_id = patient_id
        
        if not isinstance(self.comparison, bool):
            sys.exit("comparison must be true or false")
            
    def plotVitals(self, vital):
        
        if vital not in ['HR', 'RR', 'SBP', 'TEMP', 'SPO2', 'avpu']:
            sys.exit("vitals must be HR, RR, SBP, TEMP, SPO2 or avpu")

        if self.comparison == True:
            for patient_id in self.records.keys():
                plt.plot(self.records[patient_id][vital])
                plt.xlabel("Hours since first measurement")
                plt.ylabel(str(vital))
            plt.show()
        else:
            plt.plot(self.records[self.patient_id]['seconds']/3600, self.records[self.patient_id][vital], 'ro')
            plt.plot(self.records[self.patient_id]['seconds']/3600, self.records[self.patient_id][vital], 'k')
            plt.xlabel("Hours since first measurement")
            plt.ylabel(str(vital))
            plt.show()
