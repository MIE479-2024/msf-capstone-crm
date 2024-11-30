import pandas as pd
import Preprocess_WoE


def preprocess_NoWoE(file_years, labelled):
    #file_name = Preprocess_WoE.yearly_data(file_years,labelled)
    if labelled:
        processed_df_1 = Preprocess_WoE.yearly_data(file_years, labelled)
        processed_df = processed_df_1[(processed_df_1['DLQ_FLAG'] == 1) | ( processed_df_1['DLQ_FLAG'] == 0) & (processed_df_1['Ongoing'] == 1)]
        CAT_COLUMNS = [
                'PURPOSE',
                'PROP_TYPE',
                'OCC_STAT',
                'MI_TYPE',
                'FTHB_FLAG'  # not sure if this should be here
            ]
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