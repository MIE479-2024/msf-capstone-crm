import pandas as pd
import numpy as np
import pickle

def preprocess_WoE(data, labelled):
    processed_df_1= yearly_data(data, labelled)
    processed_df_1 = processed_df_1.rename(columns={'PURPOSE':'PUR', 'PROP_TYPE':'PRO','MI_TYPE':'MI','OCC_STAT':'OCC'})
    
    if labelled:
        processed_df = processed_df_1[(processed_df_1['DLQ_FLAG'] == 1) | ( processed_df_1['DLQ_FLAG'] == 0) & (processed_df_1['Ongoing'] == 1)]
        
        
        RELEVANT_COLUMNS = [
        'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO',
        'DTI', 'CSCORE_B', 'CSCORE_C', 'FTHB_FLAG', 'NUM_UNIT', 'MI_PCT', 
         'PUR', 'PRO', 'MI', 'OCC',
         'DLQ_FLAG'
            ]
        
        processed_df = processed_df[RELEVANT_COLUMNS].rename({'DLQ_FLAG': 'DLQ_90_FLAG'},axis=1)
        

        '''CAT_COLUMNS = [
            'PUR',
            'PRO',
            'OCC',
            'MI',
            'FTHB_FLAG'  # not sure if this should be here
        ]
   
    
        for col in CAT_COLUMNS:

            optb = OptimalBinning(name=col, dtype="categorical", solver="cp")
            optb.fit(processed_df[col], processed_df["DLQ_FLAG"])
            
            
            opt_bin_data = processed_df.copy()
            opt_bin_data[col] = optb.transform(opt_bin_data[col], metric="woe")'''
            
            
       
        
        '''NUMERICAL_COLUMNS = [
        "ORIG_RATE",
        "ORIG_AMOUNT",	
        "ORIG_TERM",
        "OLTV",
        "NUM_BO", 
        "DTI",
        "CSCORE_B", 
        "CSCORE_C",
        "NUM_UNIT",
        "MI_PCT"]'''
        rev_col = ["ORIG_RATE","CSCORE_B","OLTV"]


        for col in rev_col:

            
            optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
            optb.fit(processed_df[col], processed_df["DLQ_90_FLAG"])
        
            opt_bin_data = processed_df.copy()
            opt_bin_data[col] = optb.transform(processed_df[col], metric="woe")
        

        
        

        
    
    else:
        path = ""
        with open(path, "rb") as f:
            # Load pre-trained WoE mappings
            binning_models = pickle.load(f)  

        processed_df = processed_df_1.copy()

        num_col = ["ORIG_RATE","CSCORE_B","OLTV"]
        processed_df = processed_df[num_col]

        transformed_columns = {}

        for col in num_col:
            optb = binning_models[col]
            transformed_columns[col] = optb.transform(processed_df[col], metric="woe")
        opt_bin_data = pd.DataFrame(transformed_columns)
    
    
    
    
    return opt_bin_data


def yearly_data(table, labelled):
    year_table = pd.DataFrame()
    if labelled:
        year_table = preprocess(table)
    else:
        year_table = preprocess_nl(table)

    return year_table


def preprocess(table):
    
    table = table[[
        'LOAN_ID', 'orig_rt', 'orig_amt', 'orig_trm', 'oltv', 'num_bo', 'dti',
        'CSCORE_B', 'FTHB_FLG', 'purpose', 'PROP_TYP', 'NUM_UNIT', 'occ_stat',
        'mi_pct', 'CSCORE_C', 'MI_TYPE', 'ORIG_DTE', 'FRST_DTE',
        'LAST_RT', 'LAST_UPB', 'LAST_ACTIVITY_DATE', 
        'F30_DTE', 'F60_DTE', 'F90_DTE', 'F120_DTE', 'F180_DTE', 'FCE_DTE', 
        'ORIG_VAL', 'LAST_STAT', 'COMPLT_FLG', 'NET_LOSS', 'NET_SEV'
    ]].rename(columns={
        'orig_rt': 'ORIG_RATE', 'orig_amt': 'ORIG_AMOUNT', 'orig_trm': 'ORIG_TERM',
        'oltv': 'OLTV', 'num_bo': 'NUM_BO', 'dti': 'DTI', 'FTHB_FLG': 'FTHB_FLAG',
        'purpose': 'PURPOSE', 'PROP_TYP': 'PROP_TYPE', 'occ_stat': 'OCC_STAT', 
        'mi_pct': 'MI_PCT', 'LAST_RT': 'LAST_RATE', 'ORIG_VAL': 'PROP_VALUE', 
        'COMPLT_FLG': 'COMPLETE_FLAG'
    })
    
    #table['AQSN_DTE'] = pd.to_datetime(table['AQSN_DTE'])
    table['ORIG_DTE'] = pd.to_datetime(table['ORIG_DTE'])
    table['FRST_DTE'] = pd.to_datetime(table['FRST_DTE'])
    table['LAST_ACTIVITY_DATE'] = pd.to_datetime(table['LAST_ACTIVITY_DATE'])
    table['F30_DTE'] = pd.to_datetime(table['F30_DTE'])
    table['F60_DTE'] = pd.to_datetime(table['F60_DTE'])
    table['F90_DTE'] = pd.to_datetime(table['F90_DTE'])
    table['F120_DTE'] = pd.to_datetime(table['F120_DTE'])
    table['F180_DTE'] = pd.to_datetime(table['F180_DTE'])
    table['FCE_DTE'] = pd.to_datetime(table['FCE_DTE'])
        
        
    table['ORIG_RATE'] = table['ORIG_RATE'].fillna(table['ORIG_RATE'].median())
    table['DTI'] = table['DTI'].fillna(table['DTI'].median())
    table['CSCORE_B'] = table['CSCORE_B'].fillna(table['CSCORE_B'].median())
    table['CSCORE_C'] = table['CSCORE_C'].fillna(table['CSCORE_B'])
    table['MI_PCT'] = table['MI_PCT'].fillna(0)

    table['FTHB_FLAG'] = table['FTHB_FLAG'].replace({'Y': 1, 'N': 0})
    table['DLQ_FLAG'] = table['F90_DTE'].notna().astype(int)

    table['Ongoing'] = (table['LAST_STAT'] == 'C').astype(int)  
    table['Current_DLQ'] = table['LAST_STAT'].apply(lambda x: int(x) if x.isdigit() else 0) 
    table['COMPLETE_FLAG'] = table['COMPLETE_FLAG'].fillna(0)
    table['NET_LOSS'] = table['NET_LOSS'].fillna(0)
    table['NET_SEV'] = table['NET_SEV'].fillna(0)

    
    newTable = table[[
        'LOAN_ID', 'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'PROP_VALUE', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', 'CSCORE_C',
        'FTHB_FLAG', 'PURPOSE', 'PROP_TYPE','NUM_UNIT', 'OCC_STAT',  
        'MI_TYPE', 'MI_PCT',  'ORIG_DTE', 'FRST_DTE', 
        'LAST_RATE', 'LAST_UPB', 'LAST_ACTIVITY_DATE', 'DLQ_FLAG', 'Ongoing', 'Current_DLQ', 'LAST_STAT', 'COMPLETE_FLAG', 'NET_LOSS', 'NET_SEV'
    ]]

    return newTable


'''F30_DTE', 'F60_DTE', 'F90_DTE', 'F120_DTE', 'F180_DTE', 'FCE_DTE, 'COMPLT_FLG', 'NET_LOSS', 'NET_SEV'''

def preprocess_nl(table):
    
    table = table[[
        'LOAN_ID', 'orig_rt', 'orig_amt', 'orig_trm', 'oltv', 'num_bo', 'dti',
        'CSCORE_B', 'FTHB_FLG', 'purpose', 'PROP_TYP', 'NUM_UNIT', 'occ_stat',
        'mi_pct', 'CSCORE_C', 'MI_TYPE', 'ORIG_DTE', 'FRST_DTE',
        'LAST_RT', 'LAST_UPB', 'LAST_ACTIVITY_DATE', 'ORIG_VAL', 'LAST_STAT' 
    ]].rename(columns={
        'orig_rt': 'ORIG_RATE', 'orig_amt': 'ORIG_AMOUNT', 'orig_trm': 'ORIG_TERM',
        'oltv': 'OLTV', 'num_bo': 'NUM_BO', 'dti': 'DTI', 'FTHB_FLG': 'FTHB_FLAG',
        'purpose': 'PURPOSE', 'PROP_TYP': 'PROP_TYPE', 'occ_stat': 'OCC_STAT', 
        'mi_pct': 'MI_PCT', 'LAST_RT': 'LAST_RATE', 'ORIG_VAL': 'PROP_VALUE'
        
    })
    
    #table['AQSN_DTE'] = pd.to_datetime(table['AQSN_DTE'])
    table['ORIG_DTE'] = pd.to_datetime(table['ORIG_DTE'])
    table['FRST_DTE'] = pd.to_datetime(table['FRST_DTE'])
    table['LAST_ACTIVITY_DATE'] = pd.to_datetime(table['LAST_ACTIVITY_DATE'])
    
        
        
    table['ORIG_RATE'] = table['ORIG_RATE'].fillna(table['ORIG_RATE'].median())
    table['DTI'] = table['DTI'].fillna(table['DTI'].median())
    table['CSCORE_B'] = table['CSCORE_B'].fillna(table['CSCORE_B'].median())
    table['CSCORE_C'] = table['CSCORE_C'].fillna(table['CSCORE_B'])
    table['MI_PCT'] = table['MI_PCT'].fillna(0)

    table['FTHB_FLAG'] = table['FTHB_FLAG'].replace({'Y': 1, 'N': 0})
    #table['DLQ_FLAG'] = table[['F30_DTE', 'F60_DTE', 'F90_DTE', 'F120_DTE', 'F180_DTE', 'FCE_DTE']].notna().any(axis=1).astype(int)

    table['Ongoing'] = (table['LAST_STAT'] == 'C').astype(int)  
    table['Current_DLQ'] = table['LAST_STAT'].apply(lambda x: int(x) if x.isdigit() else 0) 
    

    
    newTable = table[[
        'LOAN_ID', 'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'PROP_VALUE', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', 'CSCORE_C',
        'FTHB_FLAG', 'PURPOSE', 'PROP_TYPE','NUM_UNIT', 'OCC_STAT',  
        'MI_TYPE', 'ORIG_DTE', 'FRST_DTE', 
        'LAST_RATE','MI_PCT', 'LAST_UPB', 'LAST_ACTIVITY_DATE', 'Ongoing', 'Current_DLQ', 'LAST_STAT'
    ]]

    return newTable






    lppub_column_names = ["POOL_ID", "LOAN_ID", "ACT_PERIOD", "CHANNEL", "SELLER", "SERVICER",
                      "MASTER_SERVICER", "ORIG_RATE", "CURR_RATE", "ORIG_UPB", "ISSUANCE_UPB",
                      "CURRENT_UPB", "ORIG_TERM", "ORIG_DATE", "FIRST_PAY", "LOAN_AGE",
                      "REM_MONTHS", "ADJ_REM_MONTHS", "MATR_DT", "OLTV", "OCLTV",
                      "NUM_BO", "DTI", "CSCORE_B", "CSCORE_C", "FIRST_FLAG", "PURPOSE",
                      "PROP", "NO_UNITS", "OCC_STAT", "STATE", "MSA", "ZIP", "MI_PCT",
                      "PRODUCT", "PPMT_FLG", "IO", "FIRST_PAY_IO", "MNTHS_TO_AMTZ_IO",
                      "DLQ_STATUS", "PMT_HISTORY", "MOD_FLAG", "MI_CANCEL_FLAG", "Zero_Bal_Code",
                      "ZB_DTE", "LAST_UPB", "RPRCH_DTE", "CURR_SCHD_PRNCPL", "TOT_SCHD_PRNCPL",
                      "UNSCHD_PRNCPL_CURR", "LAST_PAID_INSTALLMENT_DATE", "FORECLOSURE_DATE",
                      "DISPOSITION_DATE", "FORECLOSURE_COSTS", "PROPERTY_PRESERVATION_AND_REPAIR_COSTS",
                      "ASSET_RECOVERY_COSTS", "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS",
                      "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY", "NET_SALES_PROCEEDS",
                      "CREDIT_ENHANCEMENT_PROCEEDS", "REPURCHASES_MAKE_WHOLE_PROCEEDS",
                      "OTHER_FORECLOSURE_PROCEEDS", "NON_INTEREST_BEARING_UPB", "PRINCIPAL_FORGIVENESS_AMOUNT",
                      "ORIGINAL_LIST_START_DATE", "ORIGINAL_LIST_PRICE", "CURRENT_LIST_START_DATE",
                      "CURRENT_LIST_PRICE", "ISSUE_SCOREB", "ISSUE_SCOREC", "CURR_SCOREB",
                      "CURR_SCOREC", "MI_TYPE", "SERV_IND", "CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT",
                      "CUMULATIVE_MODIFICATION_LOSS_AMOUNT", "CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS",
                      "CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS", "HOMEREADY_PROGRAM_INDICATOR",
                      "FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT", "RELOCATION_MORTGAGE_INDICATOR",
                      "ZERO_BALANCE_CODE_CHANGE_DATE", "LOAN_HOLDBACK_INDICATOR", "LOAN_HOLDBACK_EFFECTIVE_DATE",
                      "DELINQUENT_ACCRUED_INTEREST", "PROPERTY_INSPECTION_WAIVER_INDICATOR",
                      "HIGH_BALANCE_LOAN_INDICATOR", "ARM_5_YR_INDICATOR", "ARM_PRODUCT_TYPE",
                      "MONTHS_UNTIL_FIRST_PAYMENT_RESET", "MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET",
                      "INTEREST_RATE_CHANGE_DATE", "PAYMENT_CHANGE_DATE", "ARM_INDEX",
                      "ARM_CAP_STRUCTURE", "INITIAL_INTEREST_RATE_CAP", "PERIODIC_INTEREST_RATE_CAP",
                      "LIFETIME_INTEREST_RATE_CAP", "MARGIN", "BALLOON_INDICATOR",
                      "PLAN_NUMBER", "FORBEARANCE_INDICATOR", "HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR",
                      "DEAL_NAME", "RE_PROCS_FLAG", "ADR_TYPE", "ADR_COUNT", "ADR_UPB", 
                      "PAYMENT_DEFERRAL_MOD_EVENT_FLAG", "INTEREST_BEARING_UPB"]

    lppub_column_classes = {"POOL_ID": str, "LOAN_ID": str, "ACT_PERIOD": str, "CHANNEL": str, "SELLER": str, "SERVICER": str,
                        "MASTER_SERVICER": str, "ORIG_RATE": float, "CURR_RATE": float, "ORIG_UPB": float, "ISSUANCE_UPB": float,
                        "CURRENT_UPB": float, "ORIG_TERM": "Int64", "ORIG_DATE": str, "FIRST_PAY": str, "LOAN_AGE": "Int64",
                        "REM_MONTHS": "Int64", "ADJ_REM_MONTHS": "Int64", "MATR_DT": str, "OLTV": float, "OCLTV": float,
                        "NUM_BO": "Int64", "DTI": float, "CSCORE_B": "Int64", "CSCORE_C": "Int64", "FIRST_FLAG": str, "PURPOSE": str,
                        "PROP": str, "NO_UNITS": "Int64", "OCC_STAT": str, "STATE": str, "MSA": str, "ZIP": str, "MI_PCT": float,
                        "PRODUCT": str, "PPMT_FLG": str, "IO": str, "FIRST_PAY_IO": str, "MNTHS_TO_AMTZ_IO": "Int64",
                        "DLQ_STATUS": str, "PMT_HISTORY": str, "MOD_FLAG": str, "MI_CANCEL_FLAG": str, "Zero_Bal_Code": str,
                        "ZB_DTE": str, "LAST_UPB": float, "RPRCH_DTE": str, "CURR_SCHD_PRNCPL": float, "TOT_SCHD_PRNCPL": float,
                        "UNSCHD_PRNCPL_CURR": float, "LAST_PAID_INSTALLMENT_DATE": str, "FORECLOSURE_DATE": str,
                        "DISPOSITION_DATE": str, "FORECLOSURE_COSTS": float, "PROPERTY_PRESERVATION_AND_REPAIR_COSTS": float,
                        "ASSET_RECOVERY_COSTS": float, "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS": float,
                        "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY": float, "NET_SALES_PROCEEDS": float,
                        "CREDIT_ENHANCEMENT_PROCEEDS": float, "REPURCHASES_MAKE_WHOLE_PROCEEDS": float,
                        "OTHER_FORECLOSURE_PROCEEDS": float, "NON_INTEREST_BEARING_UPB": float, "PRINCIPAL_FORGIVENESS_AMOUNT": float,
                        "ORIGINAL_LIST_START_DATE": str, "ORIGINAL_LIST_PRICE": float, "CURRENT_LIST_START_DATE": str,
                        "CURRENT_LIST_PRICE": float, "ISSUE_SCOREB": "Int64", "ISSUE_SCOREC": "Int64", "CURR_SCOREB": "Int64",
                        "CURR_SCOREC": "Int64", "MI_TYPE": str, "SERV_IND": str, "CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT": float,
                        "CUMULATIVE_MODIFICATION_LOSS_AMOUNT": float, "CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS": float,
                        "CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS": float, "HOMEREADY_PROGRAM_INDICATOR": str,
                        "FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT": float, "RELOCATION_MORTGAGE_INDICATOR": str,
                        "ZERO_BALANCE_CODE_CHANGE_DATE": str, "LOAN_HOLDBACK_INDICATOR": str, "LOAN_HOLDBACK_EFFECTIVE_DATE": str,
                        "DELINQUENT_ACCRUED_INTEREST": float, "PROPERTY_INSPECTION_WAIVER_INDICATOR": str,
                        "HIGH_BALANCE_LOAN_INDICATOR": str, "ARM_5_YR_INDICATOR": str, "ARM_PRODUCT_TYPE": str,
                        "MONTHS_UNTIL_FIRST_PAYMENT_RESET": "Int64", "MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET": "Int64",
                        "INTEREST_RATE_CHANGE_DATE": str, "PAYMENT_CHANGE_DATE": str, "ARM_INDEX": str,
                        "ARM_CAP_STRUCTURE": str, "INITIAL_INTEREST_RATE_CAP": float, "PERIODIC_INTEREST_RATE_CAP": float,
                        "LIFETIME_INTEREST_RATE_CAP": float, "MARGIN": float, "BALLOON_INDICATOR": str,
                        "PLAN_NUMBER": str, "FORBEARANCE_INDICATOR": str, "HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR": str,
                        "DEAL_NAME": str, "RE_PROCS_FLAG": str, "ADR_TYPE": str, "ADR_COUNT": "Int64", "ADR_UPB": float, 
                        "PAYMENT_DEFERRAL_MOD_EVENT_FLAG": str, "INTEREST_BEARING_UPB": float}

    
    
    
    file_number = 84
    file_year = file_number // 4
    file_year = f'{file_year:02}'
    file_year = f'20{file_year}'
    file_qtr = (file_number % 4) + 1
    file_qtr = f'Q{file_qtr}'
    file_name = f'{file_year}{file_qtr}.csv'
    print(file_name)

    chunks = pd.read_csv(file_name, delimiter='|', names=lppub_column_names, dtype=lppub_column_classes, chunksize=5000000)
    export = pd.DataFrame()


    for table in chunks:
        #print("New iteration started")
        processed = data_preparation(table, file_name)
        if export.empty:
            export = processed
        else:
            export = pd.concat([export, processed], ignore_index=True)

    export = export.drop_duplicates(subset=['LOAN_ID'], keep='last')
    path = "../dataset/"
    new_file_name = path + f"{file_year}{file_qtr}_stat.csv"
    export.to_csv(new_file_name, sep=",", na_rep="NULL", float_format='%.2f', index=False, quoting=1)