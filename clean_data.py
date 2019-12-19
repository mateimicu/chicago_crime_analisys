#!/usr/bin/env python3
import datetime

import pandas as pd


data = pd.read_csv('Crimes_-_2001_to_present.csv').fillna(0)
data = data[['Date', 'IUCR', 'Primary Type', 'Arrest', 'Location Description', 'Domestic', 'Beat', 'District', 'Ward', 'X Coordinate', 'Y Coordinate', 'Latitude', 'Longitude']]

# data[data == False] = 0.0 
data = data.applymap(lambda x: 0.0 if x == False else x)
data = data.applymap(lambda x: 1.0 if x == True else x)

def normalize(data, field_name):
    ufield = set(data[field_name])
    d = dict()

    for it in ufield:
        sz = len(d)
        d[it] = sz

    data2 = data[[field_name]]
    data[field_name] = data2.applymap(lambda x: d[x])
    return data

data = normalize(data, 'Location Description')
data = normalize(data, 'Primary Type')
data['Date'] = data[['Date']].applymap(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S %p').timestamp())

data.to_csv('dataset_clean_full.csv')
