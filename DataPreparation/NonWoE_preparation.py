import pandas as pd
import numpy as np
import sys
#sys.path.append('/Users/mengyanzhu/Documents/GitHub/msf-capstone-crm/')
#from Preprocess import preprocess_data
from sklearn.preprocessing import StandardScaler
from preprocess_general import data_preprocess

def preprocess_NoWoE(path, labelled):
    table = data_preprocess(path)
    table= preprocess_data(table, labelled).dropna()
    if labelled:
        #table = pd.read_csv(file_name, low_memory=False).dropna()

        table = table[[
            'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', # 'CSCORE_C', 
            'FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'NUM_UNIT', 
            'OCC_Principal', 'OCC_Second', 'OCC_Investor', 'MI_PCT', 'DLQ_FLAG',  'Ongoing'
            ]]

        # Definition of Bad Loans: once had a 30-day delinquency in performance history
        # Definition of Good Loans: no delinquency and continuous payments up to current
        table = table[ (table['DLQ_FLAG'] == 1) | ( (table['DLQ_FLAG'] == 0) & (table['Ongoing'] == 1) ) ]

        X = table.drop(columns=['DLQ_FLAG', 'Ongoing'])
        y = table['DLQ_FLAG'].rename({'DLQ_FLAG': 'DLQ_90_FLAG'},axis=1)

        num_col = ['ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', # 'CSCORE_C', 
        'NUM_UNIT', 'MI_PCT']
        cat_col = ['FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'OCC_Principal', 'OCC_Second', 'OCC_Investor']
        scaler = StandardScaler()
        X[num_col] = scaler.fit_transform(X[num_col])
        X = pd.concat([X[num_col], X[cat_col]], axis=1)
        data =pd.concat([X,y], axis = 1)

    else:
        table = table[[
            'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', # 'CSCORE_C', 
            'FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'NUM_UNIT', 
            'OCC_Principal', 'OCC_Second', 'OCC_Investor', 'MI_PCT' 
            ]]
        num_col = ['ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', # 'CSCORE_C', 
        'NUM_UNIT', 'MI_PCT']
        cat_col = ['FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'OCC_Principal', 'OCC_Second', 'OCC_Investor']
        scaler = StandardScaler()
        data = table.copy()
        data[num_col] = scaler.fit_transform(data[num_col])
        data = pd.concat([data[num_col], data[cat_col]], axis=1)
        
        
    
    return data






def preprocess_data(table, labelled):
    year_table = pd.DataFrame()
    #path = "/Users/mengyanzhu/Documents/GitHub/msf-capstone-crm/dataset/"
    #for file_name in i:
        #for i in range (1, 5):
            
            #file_name = path+ f"{file_year}Q{i}_stat.csv"
            #print (file_name)
    if labelled:
        quarter_table = preprocess(table)
    else:
        quarter_table = preprocess_nl(table)
    
    if year_table.empty:
        year_table = quarter_table
    else:
        year_table = pd.concat([year_table, quarter_table], ignore_index=True)


    n_row, n_col = year_table.shape
    if labelled:
        #n_dlq = sum(year_table['DLQ_FLAG'])
        print("this dataset is labelled")
        #print(f"Delinquency percentage: {round(n_dlq / n_row * 100, 2)} %")
    else:
        print("this dataset is not labelled")
    print(f"Number of rows: {n_row}")
    print(f"Number of columns: {n_col}")
    print("The total number of NA is ", sum(year_table.isna().sum()))

    duplicates = year_table['LOAN_ID'][year_table['LOAN_ID'].duplicated()]
    if not duplicates.empty:
        print(duplicates.unique())
    else:
        print("No Duplicated Loan ID")

    #output = f"{file_year}_stat.csv"
    #year_table.to_csv(output, sep=",", na_rep="NULL", float_format='%.2f', index=False, quoting=1)

    
    return year_table










### lppub_file -> table




def preprocess(table):
    
    table = table[[
        'LOAN_ID', 'orig_rt', 'orig_amt', 'orig_trm', 'oltv', 'num_bo', 'dti',
        'CSCORE_B', 'FTHB_FLG', 'purpose', 'PROP_TYP', 'NUM_UNIT', 'occ_stat',
        'mi_pct', 'CSCORE_C', 'MI_TYPE',  'ORIG_DTE', 'FRST_DTE',
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
   
    # Check Glossary #27
    table['PUR_Cash_out'] = (table['PURPOSE'] == 'C').astype(int)
    table['PUR_Refinance'] = (table['PURPOSE'] == 'R').astype(int)
    table['PUR_Purchase'] = (table['PURPOSE'] == 'P').astype(int)
    # Check Glossary #28
    table['PRO_Condominium'] = (table['PROP_TYPE'] == 'CO').astype(int)
    table['PRO_Co_operative'] = (table['PROP_TYPE'] == 'CP').astype(int)
    table['PRO_Planned_Urban'] = (table['PROP_TYPE'] == 'PU').astype(int)
    table['PRO_Manufact_Home'] = (table['PROP_TYPE'] == 'MH').astype(int)
    table['PRO_Single_Family'] = (table['PROP_TYPE'] == 'SF').astype(int)
    # Check Glossary #30
    table['OCC_Principal'] = (table['OCC_STAT'] == 'P').astype(int)
    table['OCC_Second'] = (table['OCC_STAT'] == 'S').astype(int)
    table['OCC_Investor'] = (table['OCC_STAT'] == 'I').astype(int)
    # Check Glossary #73
    table['MI_Borrower'] = (table['MI_TYPE'] == 'BPMI').astype(int)
    table['MI_Lender'] = (table['MI_TYPE'] == 'LPMI').astype(int)
    table['MI_Investor'] = (table['MI_TYPE'] == 'IPMI').astype(int)  # seems trivial

    #table['DLQ_FLAG'] = table[['F30_DTE', 'F60_DTE', 'F90_DTE', 'F120_DTE', 'F180_DTE', 'FCE_DTE']].notna().any(axis=1).astype(int)
    table['DLQ_FLAG'] = table['F90_DTE'].notna().astype(int)

    table['Ongoing'] = (table['LAST_STAT'] == 'C').astype(int)
    table['Current_DLQ'] = table['LAST_STAT'].apply(lambda x: int(x) if x.isdigit() else 0)
    table['Prepaid_Matured'] = (table['LAST_STAT'] == 'P').astype(int)
    table['Third_Party_Sale'] = (table['LAST_STAT'] == 'T').astype(int)
    table['Short_Sale'] = (table['LAST_STAT'] == 'S').astype(int)
    table['Repurchased'] = (table['LAST_STAT'] == 'R').astype(int)
    table['Deed_In_Lieu'] = (table['LAST_STAT'] == 'F').astype(int)
    table['Non_Performing_NS'] = (table['LAST_STAT'] == 'N').astype(int)
    table['Reperforming_NS'] = (table['LAST_STAT'] == 'L').astype(int)

    table['COMPLETE_FLAG'] = table['COMPLETE_FLAG'].fillna(0)
    table['NET_LOSS'] = table['NET_LOSS'].fillna(0)
    table['NET_SEV'] = table['NET_SEV'].fillna(0)

    newTable = table[[
        'LOAN_ID', 'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'PROP_VALUE', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', 'CSCORE_C',
        'FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'PRO_Condominium', 'PRO_Co_operative', 'PRO_Planned_Urban', 
        'PRO_Manufact_Home', 'PRO_Single_Family', 'NUM_UNIT', 'OCC_Principal', 'OCC_Second', 'OCC_Investor', 
        'MI_PCT', 'MI_Borrower', 'MI_Lender', 'MI_Investor',  'ORIG_DTE', 'FRST_DTE', 
        'LAST_RATE', 'LAST_UPB', 'LAST_ACTIVITY_DATE', 'DLQ_FLAG', 'Ongoing', 'Current_DLQ', 'Prepaid_Matured', 'Third_Party_Sale', 
        'Short_Sale', 'Repurchased', 'Deed_In_Lieu', 'Non_Performing_NS', 'Reperforming_NS', 'COMPLETE_FLAG', 'NET_LOSS', 'NET_SEV'
    ]]
    
    del table
    # print("The total number of NA is ", sum(newTable.isna().sum()))

    return newTable








### Assume these columns are gone
'''F30_DTE', 'F60_DTE', 'F90_DTE', 'F120_DTE', 'F180_DTE', 'FCE_DTE', 
          'COMPLT_FLG', 'NET_LOSS', 'NET_SEV'''
def preprocess_nl(table):
    
    table = table[[
        'LOAN_ID', 'orig_rt', 'orig_amt', 'orig_trm', 'oltv', 'num_bo', 'dti',
        'CSCORE_B', 'FTHB_FLG', 'purpose', 'PROP_TYP', 'NUM_UNIT', 'occ_stat',
        'mi_pct', 'CSCORE_C', 'MI_TYPE', 'ORIG_DTE', 'FRST_DTE',
        'LAST_RT', 'LAST_UPB', 'LAST_ACTIVITY_DATE', 'ORIG_VAL','LAST_STAT'
        
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
   
    # Check Glossary #27
    table['PUR_Cash_out'] = (table['PURPOSE'] == 'C').astype(int)
    table['PUR_Refinance'] = (table['PURPOSE'] == 'R').astype(int)
    table['PUR_Purchase'] = (table['PURPOSE'] == 'P').astype(int)
    # Check Glossary #28
    table['PRO_Condominium'] = (table['PROP_TYPE'] == 'CO').astype(int)
    table['PRO_Co_operative'] = (table['PROP_TYPE'] == 'CP').astype(int)
    table['PRO_Planned_Urban'] = (table['PROP_TYPE'] == 'PU').astype(int)
    table['PRO_Manufact_Home'] = (table['PROP_TYPE'] == 'MH').astype(int)
    table['PRO_Single_Family'] = (table['PROP_TYPE'] == 'SF').astype(int)
    # Check Glossary #30
    table['OCC_Principal'] = (table['OCC_STAT'] == 'P').astype(int)
    table['OCC_Second'] = (table['OCC_STAT'] == 'S').astype(int)
    table['OCC_Investor'] = (table['OCC_STAT'] == 'I').astype(int)
    # Check Glossary #73
    table['MI_Borrower'] = (table['MI_TYPE'] == 'BPMI').astype(int)
    table['MI_Lender'] = (table['MI_TYPE'] == 'LPMI').astype(int)
    table['MI_Investor'] = (table['MI_TYPE'] == 'IPMI').astype(int)  # seems trivial

    

    table['Ongoing'] = (table['LAST_STAT'] == 'C').astype(int)
    table['Current_DLQ'] = table['LAST_STAT'].apply(lambda x: int(x) if x.isdigit() else 0)
    table['Prepaid_Matured'] = (table['LAST_STAT'] == 'P').astype(int)
    table['Third_Party_Sale'] = (table['LAST_STAT'] == 'T').astype(int)
    table['Short_Sale'] = (table['LAST_STAT'] == 'S').astype(int)
    table['Repurchased'] = (table['LAST_STAT'] == 'R').astype(int)
    table['Deed_In_Lieu'] = (table['LAST_STAT'] == 'F').astype(int)
    table['Non_Performing_NS'] = (table['LAST_STAT'] == 'N').astype(int)
    table['Reperforming_NS'] = (table['LAST_STAT'] == 'L').astype(int)

   

    newTable = table[[
        'LOAN_ID', 'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'PROP_VALUE', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', 'CSCORE_C',
        'FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'PRO_Condominium', 'PRO_Co_operative', 'PRO_Planned_Urban', 
        'PRO_Manufact_Home', 'PRO_Single_Family', 'NUM_UNIT', 'OCC_Principal', 'OCC_Second', 'OCC_Investor', 
        'MI_PCT', 'MI_Borrower', 'MI_Lender', 'MI_Investor', 'ORIG_DTE', 'FRST_DTE', 
        'LAST_RATE', 'LAST_UPB', 'LAST_ACTIVITY_DATE', 'Ongoing', 'Current_DLQ', 'Prepaid_Matured', 'Third_Party_Sale', 
        'Short_Sale', 'Repurchased', 'Deed_In_Lieu', 'Non_Performing_NS', 'Reperforming_NS'
    ]]
    
    del table
    # print("The total number of NA is ", sum(newTable.isna().sum()))

    return newTable
    



