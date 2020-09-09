# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:36:03 2020

@author: rashe
"""


import pandas as pd
import numpy as np
import datetime as dt
import pickle
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


plist = pd.read_csv( '../../raw/PreVent/patient_list.csv',parse_dates=['CollectionDateTime','dob_approx'])
attendances = pd.read_csv('../../raw/PreVent/ed_attendances.csv')
wards = pd.read_csv('../../raw/PreVent/ward_stays.csv',parse_dates = ['WardStartDate', 'WardEndDate'])
input_episodes = pd.read_csv('../../raw/PreVent/inpt_episodes.csv',parse_dates=['AdmissionDate','DischargeDate','LinkedDeathdate'])
vital_signs = pd.read_csv('../../raw/PreVent/vital_signs.csv',parse_dates=['PerformedDateTime'])
inpt_diagnostic_codes =  pd.read_csv('../../raw/PreVent/inpt_diagnostic_codes.csv')
blood_gases =  pd.read_csv('../../raw/PreVent/blood_gases.csv',parse_dates=['PerformedDateTime'])
micro = pd.read_csv('../../raw/PreVent/micro.csv',parse_dates=['CollectionDateTime'])
lab_tests = pd.read_csv('../../raw/PreVent/lab_tests.csv',parse_dates=['CollectionDateTime'])
height_weight = pd.read_csv('../../raw/PreVent/height_weight.csv',parse_dates=['PerformedDateTime'])
resus = pd.read_csv('../../raw/PreVent/resus_status.csv')
resus = resus[resus.EventTag != 'DNACPR']


#compare diagcode by the microbiology test
#diag code
#covid_patients = inpt_diagnostic_codes[inpt_diagnostic_codes['DiagCode'].isin(['U071','U072'])]
#covid_patients.reset_index(inplace=True)
#covid_patients.drop(['index'],axis=1,inplace=True)

#micro table
virology = micro.copy()
virology = virology.loc[~virology['Result'].str.match(r'^;.', case=False)]
virology = virology[virology['BatTestName'] == 'SARS CORONAVIRUS-2 PCR']
virology ['numres'] = pd.to_numeric(virology['ResultFull'], errors='coerce')
virology = virology[virology['numres'].notna() | virology['ResultFull'].str.startswith('DET')]
virology.drop(['numres','ResultFull','Result'],axis=1,inplace = True)
covid_patients = pd.merge(plist, virology, on = 'AccessionNumber').dropna().drop_duplicates()
covid_patients = covid_patients[['ClusterID','CollectionDateTime_x']]
covid_patients.columns = ['ClusterID','Time']
covid_patients.sort_values(by='Time')
len(covid_patients.ClusterID.unique())



episodes = pd.merge(input_episodes, covid_patients, on=['ClusterID'])
ind = np.where(episodes.LinkedDeathdate.notna())[0]
episodes['DischargeDate'][ind] = np.NaN
#only if it was 48 hours before positive or a month after results otherwise drop the admission
episodes = episodes[(episodes.AdmissionDate - episodes.Time).astype('timedelta64[h]')> -48]
episodes = episodes[(episodes.AdmissionDate - episodes.Time).astype('timedelta64[h]')< 24 * 30]
episodes = episodes[['ClusterID','AdmissionDate','DischargeDate','LinkedDeathdate']]
episodes.drop_duplicates(keep='first',inplace=True) 
#keep patients who were addmited to hospital
patiants_id = np.intersect1d(covid_patients.ClusterID.unique(), resus.ClusterID.unique())
covid_patients = covid_patients[covid_patients['ClusterID'].isin(patiants_id)]
patiants_id = np.intersect1d(episodes.ClusterID.unique(), covid_patients.ClusterID.unique())
covid_patients = covid_patients[covid_patients['ClusterID'].isin(patiants_id)]
covid_patients.drop_duplicates(keep='first',inplace=True) 
covid_patients.sort_values(by=['ClusterID','Time'], inplace=True)
episodes.sort_values('AdmissionDate')
len(covid_patients.ClusterID.unique())


#Demographics
plist['Age'] = ((plist['CollectionDateTime']-plist['dob_approx'])/np.timedelta64(1,'Y')).astype(int)
#demos = pd.merge(attendances[['ClusterID']],plist[['ClusterID','LinkedSex','Age']], on = 'ClusterID').dropna().drop_duplicates()
demos = plist[['ClusterID','Age','LinkedSex']]
demos = pd.merge(demos,covid_patients, on=['ClusterID'])
demos.drop('Time', axis=1,inplace=True)
demos.columns = ['ClusterID','age','sex']
demos.drop_duplicates(inplace=True)

miss_class = pd.read_csv('./train_test/firstevent-composite/results/Classification/misclassified_gen_age.csv')
miss_class.columns = ['ClusterID','age','sex']
miss_class = pd.merge(attendances,miss_class,on=['ClusterID'])
age = miss_class['age'].values  
sex = miss_class['sex'].values  
etn =  miss_class['EthnicGroupCode'].values  

cor_class = pd.read_csv('./train_test/firstevent-composite/results/Classification/correctly_classified_gen_age.csv')
cor_class.columns = ['ClusterID','age','sex']
cor_class = pd.merge(cor_class,attendances,on=['ClusterID'])
cor_age = cor_class['age'].values  
cor_sex = cor_class['sex'].values  
cor_etn =  cor_class['EthnicGroupCode'].values  

print(len(cor_age[cor_age < 60]), len(cor_age[cor_age >= 60])) #25 and 30
print(len(age[age < 60]), len(age[age >= 60]))

print(len(cor_etn[cor_etn == 'A ']), len(cor_etn[cor_etn !='A '])) #28 and 32
print(len(etn[etn == 'A ']), len(etn[etn !='A '])) 

print(len(cor_sex[cor_sex == 'F']), len(cor_age[cor_sex =='M'])) #23 and 34
print(len(sex[sex =='F']), len(sex[sex=='M']))



#Vital signs
cat = ['DELIVERY DEVICE USED', 'TRACHEOSTOMY MASK MONITORING', 'AVPU SCALE', 'EYE OPENING RESPONSE', 'BEST VERBAL RESPONSE', 
      'MOTOR RESPONSE', 'BLOOD PRESSURE POSITION', 'GCS ASSESSED', 'OTHER REASONS FOR ESCALATION']

tmp = pd.merge(vital_signs,covid_patients, on=['ClusterID'],how='right')
tmp = tmp[(tmp.PerformedDateTime - tmp.Time).astype('timedelta64[h]')> -48]
tmp = tmp[(tmp.PerformedDateTime - tmp.Time).astype('timedelta64[h]')< 24 * 30]
tmp.drop('Time', axis=1,inplace=True)


tmp['Feature'] = tmp['EventName'].str.upper()
tmp.drop(['EventName','ResultUnits'], axis=1, inplace=True)
tmp.dropna(inplace=True)

tmp.sort_values(by=['ClusterID','PerformedDateTime','Feature'], inplace=True)
tmp.columns = ['ClusterID','Time','Value','Feature']
#tmp['Time'] = tmp['Time'].dt.floor('D')
tmp1 = tmp[tmp['Feature'].isin(cat)]
tmp2 = tmp[~tmp['Feature'].isin(cat)]

#??for categorical values kept the first one
tmp1.dropna(inplace=True)
tmp1 = tmp1.groupby(['ClusterID','Time','Feature'])['Value'].agg(('first')).unstack().reset_index()
tmp2['Value'] = pd.to_numeric(tmp2['Value'], errors='coerce')
tmp2 = tmp2.groupby(['ClusterID','Time','Feature'])['Value'].mean().round(2).unstack().reset_index()

for vs in tmp2.columns:
    if vs != 'Time' and vs != 'ClusterID' and vs != 'OXYGEN % DELIVERED' and vs != 'OXYGEN L/MIN DELIVERED':
        hist, bin_edges = np.histogram(tmp2[vs].dropna().values, density=True,bins =100)
        tmp2[vs].values[tmp2[vs].values > bin_edges[-5]] = bin_edges[-5]
        tmp2[vs].values[tmp2[vs].values < bin_edges[4]] = bin_edges[4]
#tmp1 = tmp1.rename({'Value': ''}, axis=1)  # new method
#tmp1.columns = [f'{i}{j}' for i, j in tmp1.columns]

vital_features = pd.merge(tmp1,tmp2, on = ['ClusterID','Time'])
vital_features = vital_features.drop(['BLOOD PRESSURE POSITION', 'GCS ASSESSED', 'OTHER REASONS FOR ESCALATION',
                                     'BEST VERBAL RESPONSE', 'EYE OPENING RESPONSE','MOTOR RESPONSE','TRACHEOSTOMY MASK MONITORING','HUMIDIFIED MONITORING'],axis =1)
#================FiO2 Calculation ======================================================
vital_features['DELIVERY DEVICE USED'].values[vital_features['DELIVERY DEVICE USED'].isna()] = 'Room Air'
vital_features["RESPIRATORY RATE"].fillna(10,inplace=True)

fixed_dev={"Room Air":21,"CPAP":100,"Other: High Flow":100,"Non-invasive system":100,"Reservoir Mask":80,}
venturi_mask={"0":21,"1":24,"2":24,"3":24,"4":28,"5":28,"6":28,"7":28,"8":35,"9":35,"10":40,"11":40,"12":60,"13":60,"14":60,"15":60}
fixed_value=["Room Air",
"CPAP",
"Other: High Flow"
,"Non-invasive system"
,"Reservoir Mask"]#, np.nan]
tmp1 = vital_features[vital_features['DELIVERY DEVICE USED'].isin(fixed_value)]
tmp1['OXYGEN % DELIVERED'] = np.vectorize(fixed_dev.get)(tmp1['DELIVERY DEVICE USED'].values)
                                                 
standard_formula=["Simple Mask","Other: Nebuliser Mask",
"Other: Tracheostomy Mask",
"Other: Oxymask"]
tmp2 = vital_features[vital_features['DELIVERY DEVICE USED'].isin(standard_formula)]
tmp2['OXYGEN % DELIVERED'] = -0.99*tmp2['RESPIRATORY RATE'] + 3.11*tmp2['OXYGEN L/MIN DELIVERED'] + 51.07

rr_formula=["Nasal cannulae"]
tmp3 = vital_features[vital_features['DELIVERY DEVICE USED'].isin(rr_formula)]
tmp3['OXYGEN % DELIVERED'] = (0.038 * tmp3['OXYGEN L/MIN DELIVERED'] + 0.208)*102.5                                               

Venturi_mask=['Venturi Face Mask']
tmp4 = vital_features[vital_features['DELIVERY DEVICE USED'].isin(Venturi_mask)]
tmp4['OXYGEN % DELIVERED']=np.vectorize(venturi_mask.get)(tmp4['OXYGEN L/MIN DELIVERED'].values.astype('int').astype('str'))

vital_features = pd.concat([tmp1,tmp2,tmp3,tmp4])
a= vital_features[vital_features['DELIVERY DEVICE USED'].isin(["Simple Mask"])]
#print(a.loc[:,['OXYGEN % DELIVERED','DELIVERY DEVICE USED','RESPIRATORY RATE','OXYGEN L/MIN DELIVERED']])
#print(a["OXYGEN % DELIVERED"].unique())
#a=vital_features.loc[:,['OXYGEN % DELIVERED','DELIVERY DEVICE USED','RESPIRATORY RATE','OXYGEN L/MIN DELIVERED']].to_csv("fio2_review.csv")
#=====================================================================================
vital_features = vital_features.drop('OXYGEN L/MIN DELIVERED', axis=1)
#=========================================================================================
#four level of oxygen support
vital_features['DELIVERY DEVICE USED'].values[vital_features['DELIVERY DEVICE USED'].values == 'Room Air'] = '0'
vital_features['DELIVERY DEVICE USED'].values[vital_features['DELIVERY DEVICE USED'].isin( 
                                              ['Simple Mask', 'Nasal Cannulae', 'Nasal cannulae', 'Venturi Face Mask', 
                                               'Other: Nebuliser Mask','Other: Oxymask', 'Other: oxymask','Other: Airvo'
                                              , 'Other: NON REBREATH MASK', 'Other: Airvo flow 40'])] = '1'
vital_features['DELIVERY DEVICE USED'].values[vital_features['DELIVERY DEVICE USED'].isin(['Other: Tracheostomy Mask',
                                              'Reservoir Mask'])] = '2'
vital_features['DELIVERY DEVICE USED'].values[vital_features['DELIVERY DEVICE USED'].isin
                                              (['High Flow','Non-invasive system', 'Other: High Flow',
                                                'CPAP','Other: high flow', 'Other: High flow','Other: high flow oxygen (AIRVO)'])] = '3'
avpu = vital_features['AVPU SCALE'].astype('category')
vital_features['AVPU SCALE'] =  avpu.cat.codes
vital_features.drop(vital_features[vital_features["DELIVERY DEVICE USED"].isna()].index.tolist(),inplace=True)


#vital_features.sort_values(['ClusterID', 'Time']).to_csv('./data_v2/vitals.csv', index=False)


#no 'DIASTOLIC BLOOD PRESSURE', 'GLASGOW COMA SCORE' for Farah' Code
vital_features_v1 = vital_features[['ClusterID', 'HEART RATE','RESPIRATORY RATE', 'SYSTOLIC BLOOD PRESSURE', 
                                 'TEMPERATURE TYMPANIC','OXYGEN SATURATION','AVPU SCALE', 'DELIVERY DEVICE USED', 'OXYGEN % DELIVERED', 'Time']]
#vital_features_v1['DELIVERY DEVICE USED'].values[vital_features['DELIVERY DEVICE USED'].isin([1,2,3])] = 1
#vital_features_v1.dropna(thresh=4,inplace=True) #at least 60 per precent of features are non nan
vital_features_v1.columns = ['ClusterID', 'HR','RR','SBP','TEMP','SPO2','avpu','masktyp','FiO2','charttime']

#includes 'DIASTOLIC BLOOD PRESSURE', 'GLASGOW COMA SCORE',
vital_features_v2 = vital_features[['ClusterID', 'HEART RATE','RESPIRATORY RATE', 'SYSTOLIC BLOOD PRESSURE', 
                                 'TEMPERATURE TYMPANIC','OXYGEN SATURATION','AVPU SCALE', 'DELIVERY DEVICE USED', 'DIASTOLIC BLOOD PRESSURE', 'GLASGOW COMA SCORE', 'OXYGEN % DELIVERED', 'Time']]
#vital_features_v2.dropna(thresh=5,inplace=True) #at least 60 per precent of features are non nan


vital_features_v3 = vital_features_v1[['ClusterID', 'HR','RR','SBP','TEMP','SPO2','avpu','FiO2','charttime']]
event = tmp[tmp['Feature'] == 'DELIVERY DEVICE USED']
event = event.groupby(['ClusterID','Time','Feature'])['Value'].agg(('first')).unstack().reset_index()
event.dropna(inplace=True)
#event = event.rename({'Value': ''}, axis=1)  # new method
#event.columns = [f'{i}{j}' for i, j in event.columns]
#event = event[event['DELIVERY DEVICE USED']!='Room Air']
event['DELIVERY DEVICE USED'].values[event['DELIVERY DEVICE USED'].values == 'Room Air'] = '0'
event['DELIVERY DEVICE USED'].values[event['DELIVERY DEVICE USED'].isin( 
                                               ['Simple Mask', 'Nasal Cannulae', 'Nasal cannulae', 'Venturi Face Mask', 
                                               'Other: Nebuliser Mask','Other: Oxymask', 'Other: oxymask','Other: Airvo'
                                              , 'Other: NON REBREATH MASK', 'Other: Airvo flow 40'])] = '1'
event['DELIVERY DEVICE USED'].values[event['DELIVERY DEVICE USED'].isin(['Other: Tracheostomy Mask',
                                              'Reservoir Mask'])] = '2'
event['DELIVERY DEVICE USED'].values[event['DELIVERY DEVICE USED'].isin
                                              (['High Flow','Non-invasive system', 'Other: High Flow',
                                                'CPAP','Other: high flow', 'Other: High flow','Other: high flow oxygen (AIRVO)'])] = '3'

event.columns = ['ClusterID', 'eventtime','next_event']
event = event.dropna().drop_duplicates()
event.reset_index(inplace=True)
event.drop(['index'],axis=1,inplace=True)


#Blood tests -- based on Thom cose
features_rename = {
    'ALBUMIN_G/L':'Albumin-g/L',
    'ALK.PHOSPHATASE_IU/L':'Alk.Phosphatase-IU/L',
    'ALT_IU/L':'ALT-IU/L',
    'BASOPHILS_X10*9/L':'Basophils-x10^9/L',
    'BILIRUBIN_UMOL/L':'Bilirubin-umol/L',
    'CREATININE_UMOL/L':'Creatinine-umol/L',
    'CRP_MG/L':'CRP-mg/L',
    'EOSINOPHILS_X10*9/L':'Eosinophils-x10^9/L',
    'HAEMATOCRIT_L/L':'Haematocrit-L/L',
    'HAEMOGLOBIN_G/L':'Haemoglobin-g/L',
    'LYMPHOCYTES_X10*9/L':'Lymphocytes-x10^9/L',
    'MEAN CELL VOL._FL':'MeanCellVol-fL',
    'MONOCYTES_X10*9/L':'Monocytes-x10^9/L',
    'NEUTROPHILS_X10*9/L':'Neutrophils-x10^9/L',
    'PLATELETS_X10*9/L':'Platelets-x10^9/L',
    'POTASSIUM_MMOL/L':'Potassium-mmol/L',
    'SODIUM_MMOL/L':'Sodium-mmol/L',
    'UREA_MMOL/L':'Urea-mmol/L',
    'WHITE CELLS_X10*9/L':'WhiteCells-x10^9/L'
}


tests = pd.merge(lab_tests,covid_patients, on=['ClusterID'])
tests = tests[(tests.CollectionDateTime - tests.Time).astype('timedelta64[h]')> -48]
tests = tests[(tests.CollectionDateTime - tests.Time).astype('timedelta64[h]')< 24 * 30]
tests.drop('Time', axis=1,inplace=True)

tests['Value'] = tests['Value'].str.upper()

#Zero low values catch-all; for any tests missed above
tests = tests.replace({'Value': r'^<.*'}, {'Value': '0'}, regex=True)

#Bilirubin NEG
tests[tests['TestName']=="BILIRUBIN"] = tests[tests['TestName']=="BILIRUBIN"].replace({'Value': r'^NEG.'}, {'Value': '0'}, regex=True)

#CRP Cleaning, 0 for <8, and 300 for >156
tests[tests['TestName']=="CRP"] = tests[tests['TestName']=="CRP"].replace({'Value': r'^>.*'}, {'Value': '300'}, regex=True)

#eGFR Cleaning
tests[tests['TestName']=="EGFR"] = tests[tests['TestName']=="EGFR"].replace({'Value': r'^>.*'}, {'Value': '150'}, regex=True)

#Potassium cleaning for >10.0 values
tests[tests['TestName']=="POTASSIUM"] = tests[tests['TestName']=="POTASSIUM"].replace({'Value': r'^>.*'}, {'Value': '10'}, regex=True)

#INR Cleaning for >10.0
tests[tests['TestName']=="INR"] = tests[tests['TestName']=="INR"].replace({'Value': r'^>.*'}, {'Value': '10'}, regex=True)
tests[tests['TestName']=="PROTHROM.INR"] = tests[tests['TestName']=="PROTHROM.INR"].replace({'Value': r'^>.*'}, {'Value': '12'}, regex=True)

#APTT Cleaning for >24.0
tests[tests['TestName']=="APTT"] = tests[tests['TestName']=="APTT"].replace({'Value': r'^>.*'}, {'Value': '24'}, regex=True)
tests[tests['TestName']=="APTT PATIENT"] = tests[tests['TestName']=="APTT PATIENT"].replace({'Value': r'^>.*'}, {'Value': '150'}, regex=True)
tests[tests['TestName']=="PROTHROMB.TIME"] = tests[tests['TestName']=="PROTHROMB.TIME"].replace({'Value': r'^>.*'}, {'Value': '160'}, regex=True)

#Urine Albumin Cleaning for >5000, <5
tests[tests['TestName']=="URINE ALBUMIN"] = tests[tests['TestName']=="URINE ALBUMIN"].replace({'Value': r'^>.*'}, {'Value': '5000'}, regex=True)

#TSH Cleaning for <0.01
tests[tests['TestName']=="TSH"] = tests[tests['TestName']=="TSH"].replace({'Value': r'^>.*'}, {'Value': '150'}, regex=True)

#Serum Ferritin Cleaning for >1560
tests[tests['TestName']=="SERUM FERRITIN"] = tests[tests['TestName']=="SERUM FERRITIN"].replace({'Value': r'^>.*'}, {'Value': '2000'}, regex=True)

#D-dimer for >20k
tests[tests['TestName']=="D-DIMER"] = tests[tests['TestName']=="D-DIMER"].replace({'Value': r'^>.*'}, {'Value': '20000'}, regex=True)

#B12 for >2000
tests[tests['TestName']=="SERUM B12"] = tests[tests['TestName']=="SERUM B12"].replace({'Value': r'^>.*'}, {'Value': '2000'}, regex=True)

#Folate for <3, and >20
tests[tests['TestName']=="SERUM FOLATE"] = tests[tests['TestName']=="SERUM FOLATE"].replace({'Value': r'^>.*'}, {'Value': '20'}, regex=True)

#Faecal Elastase >500
tests[tests['TestName']=="FAECAL ELASTASE"] = tests[tests['TestName']=="FAECAL ELASTASE"].replace({'Value': r'^>.*'}, {'Value': '1000'}, regex=True)

#Thrombin Time >240
tests[tests['TestName']=="THROMBIN TIME"] = tests[tests['TestName']=="THROMBIN TIME"].replace({'Value': r'^>.*'}, {'Value': '240'}, regex=True)

#TPO Antibodies >3000
tests[tests['TestName']=="THY PEROX AB"] = tests[tests['TestName']=="THY PEROX AB"].replace({'Value': r'^>.*'}, {'Value': '5000'}, regex=True)

#Microalbumin >400
tests[tests['TestName']=="MICROALBUMIN"] = tests[tests['TestName']=="MICROALBUMIN"].replace({'Value': r'^>.*'}, {'Value': '400'}, regex=True)

#PROCALCITONIN >100
tests[tests['TestName']=="PROCALCITONIN"] = tests[tests['TestName']=="PROCALCITONIN"].replace({'Value': r'^>.*'}, {'Value': '200'}, regex=True)

#Prothromb. Time >170
tests[tests['TestName']=="PROTHROMB. TIME"] = tests[tests['TestName']=="PROTHROMB. TIME"].replace({'Value': r'^>.*'}, {'Value': '200'}, regex=True)

#TRIGLYC >65
tests[tests['TestName']=="TRIGLYCERIDE"] = tests[tests['TestName']=="TRIGLYCERIDE"].replace({'Value': r'^>.*'}, {'Value': '65'}, regex=True)

#AST > 4202
tests[tests['TestName']=="AST"] = tests[tests['TestName']=="AST"].replace({'Value': r'^>.*'}, {'Value': '5000'}, regex=True)

#Tsat > 99
tests[tests['TestName']=="TRANSFERRIN SAT"] = tests[tests['TestName']=="TRANSFERRIN SAT"].replace({'Value': r'^>.*'}, {'Value': '100'}, regex=True)
tests[tests['TestName']=="UREA"] = tests[tests['TestName']=="UREA"].replace({'Value': r'^>.*'}, {'Value': '50'}, regex=True)

#Rename E.S.R_first to match E.S.R._first
tests.loc[tests['TestName']=="E.S.R_FIRST", 'TestName'] = "E.S.R._FIRST"

#Trop Cleaning 0 for <0.4/<0.02,>X values to 5000
tests[tests['TestName']=="TROPONIN CTNI"] = tests[tests['TestName']=="TROPONIN CTNI"].replace({'Value': r'>.*'}, {'Value': '5000'}, regex=True)

tests['Value'] = pd.to_numeric(tests['Value'], errors='coerce')
print(len(tests.ClusterID), len(tests.ClusterID.unique()))
tests.dropna(inplace=True)
print(len(tests.ClusterID), len(tests.ClusterID.unique()))

tests.loc[((tests['TestName']=='CREATININE'))&(tests['Units']=='mmol/L'), 'Value'] *= 1000 # Correct Creatinine units issue
tests.loc[((tests['TestName']=='CREATININE'))&(tests['Units']=='mmol/L'), ['TestName','Units']] = ['CREATININE','umol/L'] # Re-label lower-case test name and correctly label new units

tests.loc[(tests['TestName']=='HAEMOGLOBIN')&(tests['Units']=='g/dl'), 'Value'] *= 10 # Correct Haemoglobin units issue
tests.loc[(tests['TestName']=='HAEMOGLOBIN')&(tests['Units']=='g/dl'), 'Units'] = 'g/L' # Correctly label new units

#Troponin: Convert uG values to nG - and don't do so for values <5000
tests.loc[ (tests['TestName'].isin(['TROPONIN cTnI','POCT TROPONIN']) ) & ( tests['Units'].isin(['ug/l','microg/l']) & ( tests['Value'] < 5000) ), 'Value'] *= 1000
tests.loc[ (tests['TestName'].isin(['TROPONIN cTnI','POCT TROPONIN']) ),'Units'] = "ng/l"

#Hb conversion: multiple by 10 if Hb level is <25, as a Hb of below 25 is not comptiable with life
tests.loc[ (tests['TestName'].isin(['HAEMOGLOBIN']) ) & ( tests['Units'].isin(['g/dl',np.nan]) & ( tests['Value'] < 25) ), 'Value'] *= 10
tests.loc[ (tests['TestName'].isin(['HAEMOGLOBIN']) ),'Units'] = "g/l"

tests['Feature'] = (tests['TestName'] +'_'+ tests['Units']).str.upper()
tests.drop(['TestName','Units','RefRange'], axis=1, inplace=True)

tests.sort_values(by=['ClusterID','CollectionDateTime','Feature'], inplace=True)
tests.columns = ['ClusterID','Time','Value','Feature']

tests = tests.groupby(['ClusterID','Time','Feature'])['Value'].mean().round(2).unstack().reset_index()
tests.rename(columns=features_rename, inplace=True)
feature_list = list(features_rename.values())
blood_features = tests[['ClusterID','Time']+feature_list]
#blood_features.dropna(thresh=11,inplace=True) #at least 60 per precent of features are non nan
blood_features.columns

print(len(blood_features.index), len(blood_features.ClusterID.unique()))


bloodg = pd.merge(blood_gases,covid_patients, on=['ClusterID'])
bloodg = bloodg[(bloodg.PerformedDateTime - bloodg.Time).astype('timedelta64[h]')> -48]
bloodg = bloodg[(bloodg.PerformedDateTime - bloodg.Time).astype('timedelta64[h]')< 24 * 30]
bloodg.drop('Time', axis=1,inplace=True)

bloodg['Feature'] = bloodg['EventName'].str.upper()
bloodg.drop(['EventName','ResultUnits'], axis=1, inplace=True)

bloodg.sort_values(by=['ClusterID','PerformedDateTime','Feature'], inplace=True)
bloodg.columns = ['ClusterID','Time','Value','Feature']

bloodg['Value'] = pd.to_numeric(bloodg['Value'], errors='coerce')
bloodg= bloodg.groupby(['ClusterID','Time','Feature'])['Value'].mean().round(2).unstack().reset_index()
bloodg.drop(['SPECIMEN TYPE (BG)','PH(T)C','PCO2(T)C','PO2(T)C'], axis=1, inplace=True)
#bloodg.dropna(thresh=14,inplace=True) #at least 60 per precent of features are non nan

for vs in bloodg.columns:
    if vs != 'Time' and vs != 'ClusterID':
        hist, bin_edges = np.histogram(bloodg[vs].dropna().values, density=True,bins =100)
        bloodg[vs].values[bloodg[vs].values > bin_edges[-5]] = bin_edges[-5]
        bloodg[vs].values[bloodg[vs].values < bin_edges[4]] = bin_edges[4]
        
        
#BMI
#wnames = ['WEIGHT [UNKNOWN]', 'WEIGHT [MEASURED]', 'WEIGHT [WORKING]', 'WEIGHT ESTIMATED']

tmp = pd.merge(height_weight,covid_patients, on=['ClusterID'])
tmp = tmp[(tmp.PerformedDateTime - tmp.Time).astype('timedelta64[h]')> -48]
tmp = tmp[(tmp.PerformedDateTime - tmp.Time).astype('timedelta64[h]')< 24 * 30]
tmp.drop('Time', axis=1,inplace=True)
tmp['PerformedDateTime'] = tmp['PerformedDateTime'].dt.floor('D')

tmp['Feature'] = tmp['EventName'].str.upper()
tmp.drop(['EventName','ResultUnits'], axis=1, inplace=True)
tmp.dropna(inplace=True)
#tmp['Feature'].values[tmp['Feature'].isin(wnames)] = 'WEIGHT [MEASURED]'
tmp.sort_values(by=['ClusterID','Feature'], inplace=True)
tmp.columns = ['ClusterID','Value','Time','Feature']

tmp = tmp[tmp['Feature'].isin(['WEIGHT [MEASURED]','HEIGHT/LENGTH MEASURED'])]
tmp['Value'] = pd.to_numeric(tmp['Value'], errors='coerce')
bmi_feature = tmp.groupby(['ClusterID','Feature'])['Value'].mean().round(2).unstack().reset_index()
#bmi_feature
#bmi_feature.drop('BMI SCORE (MUST)',axis=1,inplace=True)


covid_icu = wards[wards['WardName'].isin(['J-WD Adult ICU','J-WD CTVCC'])]
covid_icu = covid_icu[['ClusterID','WardName', 'WardStartDate', 'WardEndDate']]
covid_icu = pd.merge(covid_icu,covid_patients, on = 'ClusterID',how='right')
covid_icu = covid_icu[(covid_icu.WardStartDate - covid_icu.Time).astype('timedelta64[h]')> -48]
covid_icu = covid_icu[(covid_icu.WardStartDate - covid_icu.Time).astype('timedelta64[h]')< 24 * 30]
covid_icu.drop('Time', axis=1,inplace=True)
covid_icu['next_event'] = '4'
covid_icu = covid_icu[['ClusterID','WardStartDate','next_event']]
covid_icu.columns = ['ClusterID', 'eventtime','next_event']
covid_icu = covid_icu.dropna().drop_duplicates()



discharge = episodes[['ClusterID','DischargeDate']]
discharge = discharge[discharge['DischargeDate'].notna()]
discharge['next_event'] = '5'
discharge.columns = ['ClusterID', 'eventtime','next_event']
discharge = discharge.dropna().drop_duplicates()


deaths = episodes[['ClusterID','LinkedDeathdate']]
deaths = deaths[deaths['LinkedDeathdate'].notna()]
deaths['next_event'] = '6'
deaths.columns = ['ClusterID', 'eventtime','next_event']
deaths = deaths.dropna().drop_duplicates()


events_2 = discharge.append(covid_icu,ignore_index=True)
events_2 = events_2.append(deaths,ignore_index=True)

events_all = pd.concat([event,covid_icu])
events_all['next_event'].unique()

#discharge,icu,death
data = pd.merge(vital_features_v1,episodes[['ClusterID','AdmissionDate','DischargeDate']], on = ['ClusterID'])
data = pd.merge(data,demos, on = ['ClusterID'])
data = pd.merge(data,events_2, on = ['ClusterID'])
data['hrs_to_firstevent'] = (data.eventtime - data.charttime).astype('timedelta64[h]')
data['prev_event']=np.where(data.hrs_to_firstevent<0, 1,0)
data = data[['ClusterID','HR','RR','SBP','TEMP','SPO2','avpu','masktyp','AdmissionDate','DischargeDate',
         'age','sex','eventtime','next_event','charttime','hrs_to_firstevent','prev_event']]
data.columns = ['hadm_id','HR','RR','SBP','TEMP','SPO2','avpu','masktyp','admittime','dischtime',
                'age','sex','eventtime','next_event','charttime','hrs_to_firstevent','prev_event']
data.sort_values(['hadm_id', 'charttime']).to_csv('./data_v2/event_obs.csv', index=False)

#stratified based on event
X_train, X_test, _, _ = train_test_split(data, data[['next_event']].values, test_size=0.3, random_state=42)
X_train.sort_values(['hadm_id', 'charttime']).to_csv('./train_test/train_event_obs.csv', index=False)
X_test.sort_values(['hadm_id', 'charttime']).to_csv('./train_test/test_event_obs.csv', index=False)

vital_features_v2.sort_values(['ClusterID', 'Time']).to_csv('./data_v2/vitals.csv', index=False)
blood_features.sort_values(['ClusterID', 'Time']).to_csv('./data_v2/blood_tests.csv', index=False)
bloodg.sort_values(['ClusterID', 'Time']).to_csv('./data_v2/blood_tests.csv', index=False)
bmi_feature.sort_values(['ClusterID']).to_csv('./data_v2/BMI.csv', index=False)
covid_patients.sort_values(['ClusterID', 'Time']).to_csv('./data_v2/covid_patients.csv', index=False)
demos.sort_values(['ClusterID']).to_csv('./data_v2/demos.csv', index=False)
covid_icu[['ClusterID','eventtime']].sort_values(['ClusterID', 'eventtime']).to_csv('./data_v2/icuadmissions.csv', index=False)
episodes.sort_values(['ClusterID','AdmissionDate']).to_csv('./data_v2/episodes.csv', index=False)


#care level vital signs
data = pd.merge(vital_features_v3,episodes[['ClusterID','AdmissionDate','DischargeDate']], on = ['ClusterID'])
data = pd.merge(data,demos, on = ['ClusterID'])
#data = pd.merge(data,bmi_feature,on = ['ClusterID'] )
data = pd.merge(data,event, on = ['ClusterID'])
data['hrs_to_firstevent'] = (data.eventtime - data.charttime).astype('timedelta64[h]')
data['prev_event']=np.where(data.hrs_to_firstevent<0, 1,0)
data = data[['ClusterID','HR','RR','SBP','TEMP','SPO2','avpu','AdmissionDate','DischargeDate',
         'age','sex','eventtime','next_event','charttime','hrs_to_firstevent','prev_event']]
data.columns = ['hadm_id','HR','RR','SBP','TEMP','SPO2','avpu','admittime','dischtime',
                'age','sex','eventtime','next_event','charttime','hrs_to_firstevent','prev_event']
#care level vital signs
data.sort_values(['hadm_id', 'charttime']).to_csv('./data_v2/care_event_obs_vital.csv', index=False)

#stratified based on event
X_train, X_test, _, _ = train_test_split(data, data[['next_event']].values, test_size=0.3, random_state=42)
X_train.sort_values(['hadm_id', 'charttime']).to_csv('./train_test/train_care_event_vitals.csv', index=False)
X_test.sort_values(['hadm_id', 'charttime']).to_csv('./train_test/test_care_event_vitals.csv', index=False)


#all evennts vital signs
data = pd.merge(vital_features_v1,episodes[['ClusterID','AdmissionDate','DischargeDate']], on = ['ClusterID'])
data = pd.merge(data,demos, on = ['ClusterID'])
#data = pd.merge(data,bmi_feature,on = ['ClusterID'] )
data = pd.merge(data,events_all, on = ['ClusterID'])
data[['eventtime', 'charttime']] = data[['eventtime', 'charttime']].apply(pd.to_datetime, errors='coerce')
data['hrs_to_firstevent'] = (data.eventtime - data.charttime).astype('timedelta64[h]')
data['prev_event']=np.where(data.hrs_to_firstevent<0, 1,0)
data = data[['ClusterID','HR','RR','SBP','TEMP','SPO2','avpu','masktyp','AdmissionDate','DischargeDate',
         'age','sex','eventtime','next_event','charttime','hrs_to_firstevent','prev_event']]
data.columns = ['hadm_id','HR','RR','SBP','TEMP','SPO2','avpu','masktyp','admittime','dischtime',
                'age','sex','eventtime','next_event','charttime','hrs_to_firstevent','prev_event']
data.drop_duplicates(inplace=True)
#care level vital signs
data.sort_values(['hadm_id', 'charttime']).to_csv('./data_v2/all_event_obs.csv', index=False)

#stratified based on event
#X_train, X_test, _, _ = train_test_split(data, data[['next_event']].values, test_size=0.3, random_state=42)
#X_train.sort_values(['hadm_id', 'charttime']).to_csv('./train_test/allevents_train_event_vitals.csv', index=False)
#X_test.sort_values(['hadm_id', 'charttime']).to_csv('./train_test/allevents_test_event_vitals.csv', index=False)


#all evennts all features
vital_features_v1.columns = np.concatenate((['ClusterID'],['Vital_Signs ' + x for x in vital_features_v1.columns[1:-1]],['charttime']))
blood_features.columns = np.concatenate((['ClusterID','charttime'],['Blood_Test ' + x for x in blood_features.columns[2:]]))
bloodg.columns = np.concatenate((['ClusterID','charttime'],['Blood_Gas ' + x for x in bloodg.columns[2:]]))

bf = blood_features.sort_values(by='charttime').reset_index()
bg = bloodg.sort_values(by='charttime')
vf = vital_features_v1.sort_values(by='charttime')

bf['hrs_from_adm_lab'] = np.NaN
for i in range(len(bf)):
    tmp = vf[vf['ClusterID']== bf.loc[i,'ClusterID']]
    tmp = tmp[ (bf.loc[i,'charttime']-tmp['charttime']).astype('timedelta64[h]') >=0]
    if len(tmp) >0:
        tmp['hrs_from_adm_lab'] = (bf.loc[i,'charttime']-tmp['charttime']).astype('timedelta64[h]')
        tmp = tmp.sort_values(by='hrs_from_adm_lab').reset_index()
   #     print(tmp.loc[0,'hrs_from_adm_lab'],bf.loc[i,'charttime'])
        if tmp.loc[0,'hrs_from_adm_lab'] < 120:
            bf.loc[i,'hrs_from_adm_lab'] = tmp.loc[0,'hrs_from_adm_lab']
        else:
            bf.loc[i,'hrs_from_adm_lab'] = 120
    else: 
        bf.loc[i,'hrs_from_adm_lab'] = 120     
        
        
data = pd.merge(bf,episodes[['ClusterID','AdmissionDate','DischargeDate']], on = ['ClusterID'])
data = pd.merge(data,demos, on = ['ClusterID'])

data = pd.merge(data,events_all, on = ['ClusterID'])

data[['eventtime', 'charttime']] = data[['eventtime', 'charttime']].apply(pd.to_datetime, errors='coerce')
data['hrs_to_firstevent'] = (data.eventtime - data.charttime).astype('timedelta64[h]')
data['prev_event']=np.where(data.hrs_to_firstevent<0, 1,0)

cnames = np.concatenate((['ClusterID'],np.array(blood_features.columns[2:]),
                        ['AdmissionDate','DischargeDate','age','sex','eventtime','next_event','charttime',
                         'hrs_to_firstevent','prev_event','hrs_from_adm_lab']))
data = data[cnames]

data.columns = np.concatenate((['hadm_id'],np.array(blood_features.columns[2:]),
                               ['admittime','dischtime','age','sex','eventtime','next_event'
                                ,'charttime','hrs_to_firstevent','prev_event','hrs_from_adm_lab']))

data.drop_duplicates(inplace=True)
#care level vital signs

data.sort_values(['hadm_id', 'charttime']).to_csv('./data_v2/all_event_Blood.csv', index=False)


data = pd.merge(bg,episodes[['ClusterID','AdmissionDate','DischargeDate']], on = ['ClusterID'])
data = pd.merge(data,demos, on = ['ClusterID'])

data = pd.merge(data,events_all, on = ['ClusterID'])

data[['eventtime', 'charttime']] = data[['eventtime', 'charttime']].apply(pd.to_datetime, errors='coerce')
data['hrs_to_firstevent'] = (data.eventtime - data.charttime).astype('timedelta64[h]')
data['prev_event']=np.where(data.hrs_to_firstevent<0, 1,0)

cnames = np.concatenate((['ClusterID'],np.array(bloodg.columns[2:]),
                        ['AdmissionDate','DischargeDate','age','sex','eventtime','next_event','charttime',
                         'hrs_to_firstevent','prev_event']))
data = data[cnames]

data.columns = np.concatenate((['hadm_id'],np.array(bloodg.columns[2:]),
                               ['admittime','dischtime','age','sex','eventtime','next_event'
                                ,'charttime','hrs_to_firstevent','prev_event']))

data.drop_duplicates(inplace=True)
#care level vital signs

data.sort_values(['hadm_id', 'charttime']).to_csv('./data_v2/all_event_BloodGas.csv', index=False)


def event_feature_label_save(inp,name, N=24, W=6):
    data = inp.copy()
    data = data[data['prev_event']==0]
#    tmp = data[['hadm_id','prev_event','admittime']]
#    tmp.drop_duplicates(inplace=True)
#    tmp = tmp[tmp['prev_event']>1]
#    print(tmp)
    data = data[data['next_event'] >1] #remove care l0 and l1 as event
    data = data[data['hrs_to_firstevent']>1]
    data['label'] = np.where((data['hrs_to_firstevent']<=N),1,0)
  #  data['label'] =data['next_event']
  #  data.loc[data['hrs_to_firstevent']>N,'label'] = 0
    
    data = data.sort_values(by=['hadm_id','hrs_to_firstevent']).groupby(['hadm_id','charttime'],as_index=False).first()
    tmp = data[data['label']>=1]
    tmp = tmp.sort_values(by=['hadm_id','hrs_to_firstevent']).groupby(['hadm_id','label'],as_index=False).first()
    data = pd.concat([data[data['label']==0],tmp])
  #  data.loc[data['label']>=1,'label']=1
    
    BloodTest_cols_bool = data.columns.str.contains('Blood_Test')
    BloodGas_cols_bool = data.columns.str.contains('Blood_Gas')
    VitalSign_cols_bool = data.columns.str.contains('Vital_Sign')
    
    BloodTest_features = data.loc[:,BloodTest_cols_bool]
    BloodGas_features = data.loc[:,BloodGas_cols_bool]
    VitalSign_features = data.loc[:,VitalSign_cols_bool]
    other_features = data.loc[:,['hadm_id','label','age','sex','hrs_to_firstevent', 'hrs_from_adm_lab']]
    
    X = pd.concat((other_features,VitalSign_features, BloodTest_features, BloodGas_features),1)
    
    X.dropna(thresh=len(X.columns)*0.6,inplace=True)
    
    delta_features = np.empty((len(data),3*len(VitalSign_features.columns)))
    cnames = VitalSign_features.columns
    time = data['charttime'].values
    ids = data['hadm_id'].values
    nn= len(VitalSign_features.columns)
    data.reset_index(inplace = True)
    for i in range(len(data)):
        time1 = time[i]
        tmp=inp[inp['hadm_id']==ids[i]]
        tmp = tmp[(time1 - tmp['charttime']).astype('timedelta64[h]') >0]
        tmp = tmp[(time1 - tmp['charttime']).astype('timedelta64[h]') <=W]
        tmp = tmp.loc[:,cnames].dropna()
        if len(tmp) > 0: 
            delta_features[i,:nn] = tmp[cnames].mean(0).values
            delta_features[i,nn:2*nn] = tmp[cnames].max(0).values - tmp[cnames].min(0).values
            delta_features[i,2*nn:3*nn] = data.loc[i,cnames] - tmp[cnames].mean(0).values
        else:
            delta_features[i,:nn*3] = np.NaN
    n1 = ['Var_Mean_' + x for x in cnames]
    n2 = ['Max_Min_' + x for x in cnames]
    n3 = ['Delta_Mean_' + x for x in cnames]
    
    deltanames = np.concatenate((n1,n2,n3))#,n4,n5))
    
    delta_df = pd.DataFrame(data=delta_features, columns = deltanames)
    data_event = pd.concat([X, delta_df.reindex(X.index)], axis=1)
    
    data_event.to_csv('./train_test/firstevent-composite/'+name, index=False)
    return data_event


features = pd.merge_asof(bg, bf, by='ClusterID', left_on='charttime', right_on='charttime', direction='nearest', tolerance=pd.Timedelta(24*5, unit='h'))
features = pd.merge_asof(vf,features, by='ClusterID', left_on='charttime', right_on='charttime', direction='nearest', tolerance=pd.Timedelta(24*5, unit='h'))

data = pd.merge(features,episodes[['ClusterID','AdmissionDate','DischargeDate']], on = ['ClusterID'])
data = pd.merge(data,demos, on = ['ClusterID'])

data = pd.merge(data,events_all, on = ['ClusterID'])

data[['eventtime', 'charttime']] = data[['eventtime', 'charttime']].apply(pd.to_datetime, errors='coerce')
data['hrs_to_firstevent'] = (data.eventtime - data.charttime).astype('timedelta64[h]')
data['prev_event']=np.where(data.hrs_to_firstevent<0, 1,0)

cnames = np.concatenate((['ClusterID'],np.array(vital_features_v1.columns[1:-1]),np.array(blood_features.columns[2:])
                         ,np.array(bloodg.columns[2:]),
                        ['AdmissionDate','DischargeDate','age','sex','eventtime','next_event','charttime',
                         'hrs_to_firstevent','prev_event','hrs_from_adm_lab']))
data = data[cnames]

data.columns = np.concatenate((['hadm_id'],np.array(vital_features_v1.columns[1:-1]),np.array(blood_features.columns[2:])
                               ,np.array(bloodg.columns[2:]),['admittime','dischtime','age','sex','eventtime','next_event'
                                                              ,'charttime','hrs_to_firstevent','prev_event','hrs_from_adm_lab']))


data.drop_duplicates(inplace=True)
#care level vital signs
data.sort_values(['hadm_id', 'charttime']).to_csv('./data_v2/all_event_features.csv', index=False)

print(len(data.index),len(data.hadm_id.unique()))

data = pd.read_csv('./data_v2/all_event_features.csv',parse_dates = ['charttime','admittime'])
N = [24]#[6,12,24]
W=[24]#[6,12,24]
#####save the data
for n in N:
    for w in W:
        data = event_feature_label_save(data,'dataevents_updated2_N'+str(n)+'_W'+str(w)+'.csv', n,w)    
        print(len(data.index),len(data.hadm_id.unique()))
#stratified based on event
#X_train, X_test, _, _ = train_test_split(data, data[['next_event']].values, test_size=0.3, random_state=42)
#X_train.sort_values(['hadm_id', 'charttime']).to_csv('./train_test/allevents_train_event_all_features.csv', index=False)
#X_test.sort_values(['hadm_id', 'charttime']).to_csv('./train_test/allevents_test_event_all_features.csv', index=False)
        
        
        
#care level blood tests
data = pd.merge(blood_features,episodes[['ClusterID','AdmissionDate','DischargeDate']], on = ['ClusterID'])
data = pd.merge(data,demos, on = ['ClusterID'])
#data = pd.merge(data,bmi_feature,on = ['ClusterID'] )
data = pd.merge(data,event, on = ['ClusterID'])
data.rename(columns={'Time':'charttime'},inplace=True)
data['hrs_to_firstevent'] = (data.eventtime - data.charttime).astype('timedelta64[h]')
data['prev_event']=np.where(data.hrs_to_firstevent<0, 1,0)
data.sort_values(['ClusterID', 'charttime']).to_csv('./data_v2/care_event_obs_bloodtests.csv', index=False)


#care level bloodg
data = pd.merge(bloodg,episodes[['ClusterID','AdmissionDate','DischargeDate']], on = ['ClusterID'])
data = pd.merge(data,demos, on = ['ClusterID'])
#data = pd.merge(data,bmi_feature,on = ['ClusterID'] )
data = pd.merge(data,event, on = ['ClusterID'])
data.rename(columns={'Time':'charttime'},inplace=True)
data['hrs_to_firstevent'] = (data.eventtime - data.charttime).astype('timedelta64[h]')
data['prev_event']=np.where(data.hrs_to_firstevent<0, 1,0)
data.sort_values(['ClusterID', 'charttime']).to_csv('./data_v2/care_event_obs_bloodg.csv', index=False)



#train-test split by ClusterID
#(1) based on tine of positive test
test = covid_patients[((pd.DatetimeIndex(covid_patients['Time']).month==4) &(pd.DatetimeIndex(covid_patients['Time']).
                                day >=15))].sort_values(['ClusterID', 'Time'])
train = covid_patients[~covid_patients['ClusterID'].isin(test['ClusterID'])]
train = train[['ClusterID']].drop_duplicates()
test = test[['ClusterID']].drop_duplicates()
train.to_csv('./train_test/train_time.csv', index=False)
test.to_csv('./train_test/test_time.csv', index=False)

#distirbuted equally across time
test_train = covid_patients.sort_values(['Time'])
test_train = test_train[['ClusterID']].drop_duplicates()
test = test_train.iloc[::3, :]
train = test_train[~test_train['ClusterID'].isin(test['ClusterID'])]
train.to_csv('./train_test/train_eqd.csv', index=False)
test.to_csv('./train_test/test_eqd.csv', index=False)