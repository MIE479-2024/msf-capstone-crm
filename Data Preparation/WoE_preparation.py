import pandas as pd
import Preprocess


def preprocess_NoWoE(file_year, labelled):
    file_name = Preprocess.preprocess_data(file_year, False, labelled)
    if labelled:
        table = pd.read_csv(file_name, low_memory=False).dropna()

        table = table[[
            'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', # 'CSCORE_C', 
            'FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'NUM_UNIT', 
            'OCC_Principal', 'OCC_Second', 'OCC_Investor', 'MI_PCT', 'DLQ_FLAG', 'Ongoing', 
            'Current_DLQ', 'Prepaid_Matured']]

        # Definition of Bad Loans: once had a 30-day delinquency in performance history
        # Definition of Good Loans: no delinquency and continuous payments up to current
        table = table[ (table['DLQ_FLAG'] == 1) | ( (table['DLQ_FLAG'] == 0) & (table['Ongoing'] == 1) ) ]

        X = table.drop(columns=['DLQ_FLAG', 'Ongoing', 'Current_DLQ', 'Prepaid_Matured'])
        y = table['DLQ_FLAG']

        num_col = ['ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', # 'CSCORE_C', 
        'NUM_UNIT', 'MI_PCT']
        cat_col = ['FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'OCC_Principal', 'OCC_Second', 'OCC_Investor']
       
        X[num_col] = scaler.fit_transform(X[num_col])
        X = pd.concat([X[num_col], X[cat_col]], axis=1)

    else:
        pass
        
        '''table = pd.read_csv(file_name, low_memory=False).dropna()

        table = table[[
            'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', # 'CSCORE_C', 
            'FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'NUM_UNIT', 
            'OCC_Principal', 'OCC_Second', 'OCC_Investor', 'MI_PCT', 'DLQ_FLAG', 'Ongoing', 
            'Current_DLQ', 'Prepaid_Matured']]

        # Definition of Bad Loans: once had a 30-day delinquency in performance history
        # Definition of Good Loans: no delinquency and continuous payments up to current
        table = table[ (table['DLQ_FLAG'] == 1) | ( (table['DLQ_FLAG'] == 0) & (table['Ongoing'] == 1) ) ]

        X = table.drop(columns=['DLQ_FLAG', 'Ongoing', 'Current_DLQ', 'Prepaid_Matured'])
        y = []

        num_col = ['ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', # 'CSCORE_C', 
           'NUM_UNIT', 'MI_PCT']
        cat_col = ['FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'OCC_Principal', 'OCC_Second', 'OCC_Investor']
        scaler = StandardScaler()
        X[num_col] = scaler.fit_transform(X[num_col])
        X = pd.concat([X[num_col], X[cat_col]], axis=1)'''
    
    return X, y 