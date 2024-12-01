import pandas as pd
import numpy as np
import pickle

WOE_COLUMNS = ["ORIG_RATE","CSCORE_B","OLTV"]

def preprocess_WoE(data):
    data = filter_woe_columns(data)
    path = "models/binning_models.pkl"
    with open(path, "rb") as f:
        binning_models = pickle.load(f)  

    transformed_columns = {}

    for col in WOE_COLUMNS:
        optb = binning_models[col]
        transformed_columns[col] = optb.transform(data[col], metric="woe")
    opt_bin_data = pd.DataFrame(transformed_columns)

    return opt_bin_data

def filter_woe_columns(table):
    # TODO: add empty warnings
    table = table[[
        'LOAN_ID', 'orig_rt', 'oltv', 'CSCORE_B', 'F90_DTE', 'LAST_STAT'
    ]].rename(columns={
        'orig_rt': 'ORIG_RATE', 'oltv': 'OLTV'
    })
    table['F90_DTE'] = pd.to_datetime(table['F90_DTE'])
    table['ORIG_RATE'] = table['ORIG_RATE'].fillna(table['ORIG_RATE'].median())
    table['CSCORE_B'] = table['CSCORE_B'].fillna(table['CSCORE_B'].median())
    table['OLTV'] = table['OLTV'].fillna(table['OLTV'].median())
    table['DLQ_FLAG'] = table['F90_DTE'].notna().astype(int)
    table['Ongoing'] = (table['LAST_STAT'] == 'C').astype(int)  
    newTable = table[ ['LOAN_ID'] + WOE_COLUMNS + ['DLQ_FLAG', 'Ongoing']]
    return newTable
