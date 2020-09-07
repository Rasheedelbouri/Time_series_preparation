#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:28:13 2020

@author: kebl4170
"""

import os
import pandas as pd

path = '../../../netshares/ibme/Projects_1/orchid/processing/SK/data_v2'
            
vitals = pd.read_csv(os.path.join(path, "vitals.csv"))