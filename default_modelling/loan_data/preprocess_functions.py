import pandas as pd

def rename_cols(table):
    table = table[[
        'LOAN_ID', 'orig_rt', 'orig_amt', 'orig_trm', 'ocltv', 'num_bo', 'dti',
        'CSCORE_B', 'FTHB_FLG', 'purpose', 'PROP_TYP', 'NUM_UNIT', 'occ_stat',
        'mi_pct', 'CSCORE_C', 'MI_TYPE', 'AQSN_DTE', 'ORIG_DTE', 'FRST_DTE',
        'F30_DTE', 'F60_DTE', 'F90_DTE', 'F120_DTE', 'F180_DTE', 'FCE_DTE'
    ]].rename(columns={
        'orig_rt': 'ORIG_RATE', 'orig_amt': 'ORIG_AMOUNT', 'orig_trm': 'ORIG_TERM',
        'ocltv': 'OCLTV', 'num_bo': 'NUM_BO', 'dti': 'DTI', 'FTHB_FLG': 'FTHB_FLAG',
        'purpose': 'PURPOSE', 'PROP_TYP': 'PROP_TYPE', 'occ_stat': 'OCC_STAT', 'mi_pct': 'MI_PCT'
    })
    return table


def fix_datetimes(table):

    table['AQSN_DTE'] = pd.to_datetime(table['AQSN_DTE'])
    table['ORIG_DTE'] = pd.to_datetime(table['ORIG_DTE'])
    table['FRST_DTE'] = pd.to_datetime(table['FRST_DTE'])
    table['F30_DTE'] = pd.to_datetime(table['F30_DTE'])
    table['F60_DTE'] = pd.to_datetime(table['F60_DTE'])
    table['F90_DTE'] = pd.to_datetime(table['F90_DTE'])
    table['F120_DTE'] = pd.to_datetime(table['F120_DTE'])
    table['F180_DTE'] = pd.to_datetime(table['F180_DTE'])
    table['FCE_DTE'] = pd.to_datetime(table['FCE_DTE'])

    return table 


def one_hot_encoding(table):
    # Check Glossary #27
    bt2 = table
    bt2['PUR_Cash_out'] = (bt2['PURPOSE'] == 'C').astype(int)
    bt2['PUR_Refinance'] = (bt2['PURPOSE'] == 'R').astype(int)
    bt2['PUR_Purchase'] = (bt2['PURPOSE'] == 'P').astype(int)
    # Check Glossary #28
    bt2['PRO_Condominium'] = (bt2['PROP_TYPE'] == 'CO').astype(int)
    bt2['PRO_Co_operative'] = (bt2['PROP_TYPE'] == 'CP').astype(int)
    bt2['PRO_Planned_Urban'] = (bt2['PROP_TYPE'] == 'PU').astype(int)
    bt2['PRO_Manufact_Home'] = (bt2['PROP_TYPE'] == 'MH').astype(int)
    bt2['PRO_Single_Family'] = (bt2['PROP_TYPE'] == 'SF').astype(int)
    # Check Glossary #30
    bt2['OCC_Principal'] = (bt2['OCC_STAT'] == 'P').astype(int)
    bt2['OCC_Second'] = (bt2['OCC_STAT'] == 'S').astype(int)
    bt2['OCC_Investor'] = (bt2['OCC_STAT'] == 'I').astype(int)
    # Check Glossary #73
    bt2['MI_Borrower'] = (bt2['MI_TYPE'] == 'BPMI').astype(int)
    bt2['MI_Lender'] = (bt2['MI_TYPE'] == 'LPMI').astype(int)
    bt2['MI_Investor'] = (bt2['MI_TYPE'] == 'IPMI').astype(int)  # seems trivial
    return table


def filter_origination_date(table):
    # Origination date is at most two months earlier than acquisition date
    table['date_diff'] = (table['AQSN_DTE'] - table['ORIG_DTE']).dt.days
    table = table[table['date_diff'] <= 70]
    return table


def fill_missing_values(table):
    table['MI_PCT'] = table['MI_PCT'].fillna(0)
    table['CSCORE_C'] = table['CSCORE_C'].fillna(table['CSCORE_B'])
    return table


def deliquency_flag(table):
    table['DLQ_FLAG'] = table[['F30_DTE', 'F60_DTE', 'F90_DTE', 'F120_DTE', 'F180_DTE', 'FCE_DTE']].notna().any(axis=1).astype(int)
    return table

def filter_output_cols(table, cat_encoding):
    
    if not cat_encoding:
        output_cols = [
        'LOAN_ID', 'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OCLTV', 'NUM_BO', 'DTI', 'CSCORE_B', 'CSCORE_C',
        'FTHB_FLAG', 'PURPOSE', 'PROP_TYPE', 'NUM_UNIT', 'OCC_STAT', 'MI_PCT', 'MI_TYPE', 'AQSN_DTE', 'ORIG_DTE', 'FRST_DTE',
        'DLQ_FLAG'
        ]
    else:
        output_cols = output_cols = [
        'LOAN_ID', 'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OCLTV', 'NUM_BO', 'DTI', 'CSCORE_B', 'CSCORE_C',
        'FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'PRO_Condominium', 'PRO_Co_operative',
        'PRO_Planned_Urban', 'PRO_Manufact_Home', 'PRO_Single_Family', 'NUM_UNIT', 'OCC_Principal', 'OCC_Second', 
        'OCC_Investor', 'MI_PCT', 'MI_Borrower', 'MI_Lender', 'MI_Investor', 'AQSN_DTE', 'ORIG_DTE', 'FRST_DTE',
        'DLQ_FLAG'
        ]

    return table[output_cols]

def preprocess(table, cat_encoding = False):
    
    table = rename_cols(table)
    table = fix_datetimes(table)
    

    if cat_encoding:
        table = one_hot_encoding(table)

    table = fill_missing_values(table)
    table = deliquency_flag(table)
    table = filter_output_cols(table, cat_encoding)
    
    n_row, n_col = table.shape
    print(f"Number of rows: {n_row}")
    print(f"Number of columns: {n_col}")


    return table