import pandas as pd
import numpy as np
import pickle
import os
import pkgutil


WOE_COLUMNS = ["ORIG_RATE","CSCORE_B","OLTV"]


def load_model(model_name):
    """Loads pre-trained model from file."""
    model_data = pkgutil.get_data('utils', f'models/{model_name}')
    return pickle.loads(model_data)


def preprocess_WoE(data, labelled):
    """
    Preprocess the input data using Weight of Evidence (WoE) encoding.

    Args:
        data (pd.DataFrame): The input loan dataset.
        labelled (bool): Indicates if the dataset includes labels for evaluation.

    Returns:
        pd.DataFrame: The preprocessed dataset with WoE-encoded columns.

    Raises:
        ValueError: If any required column is missing or completely empty.
    """
    data = filter_woe_columns(data)
    data = data.astype(float)

    if labelled:
        data = data[ (data['DLQ_90_FLAG'] == 1) | ( (data['DLQ_90_FLAG'] == 0) & (data['Ongoing'] == 1) ) ]

    binning_models = load_model('binning_models.pkl')

    transformed_columns = {}

    for col in WOE_COLUMNS:
        optb = binning_models[col]
        transformed_columns[col] = optb.transform(data[col], metric="woe")
    opt_bin_data = pd.DataFrame(transformed_columns, index=data.index)
    opt_bin_data["DLQ_90_FLAG"] = data["DLQ_90_FLAG"]
    return opt_bin_data


def filter_woe_columns(table):
    """
    Filter and preprocess columns required for WoE transformation.

    Args:
        table (pd.DataFrame): The raw input loan dataset.

    Returns:
        pd.DataFrame: The filtered and preprocessed dataset for WoE encoding.

    Raises:
        ValueError: If any required column is completely empty.
    """
    for column in ['orig_rt', 'oltv', 'CSCORE_B']:
        if table[column].isna().all():
            raise ValueError(f"The column '{column}' is completely empty, but required for classification models. Please provide valid data.")

    table = table[[
        'orig_rt', 'oltv', 'CSCORE_B', 'F90_DTE', 'LAST_STAT'
    ]].rename(columns={
        'orig_rt': 'ORIG_RATE', 'oltv': 'OLTV'
    })
    table['F90_DTE'] = pd.to_datetime(table['F90_DTE'])
    table['ORIG_RATE'] = table['ORIG_RATE'].fillna(table['ORIG_RATE'].median())
    table['CSCORE_B'] = table['CSCORE_B'].fillna(table['CSCORE_B'].median())
    table['OLTV'] = table['OLTV'].fillna(table['OLTV'].median())
    table['DLQ_90_FLAG'] = table['F90_DTE'].notna().astype(int)
    table['Ongoing'] = (table['LAST_STAT'] == 'C').astype(int)  
    newTable = table[ WOE_COLUMNS + ['DLQ_90_FLAG', 'Ongoing']]
    return newTable
