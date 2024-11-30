import pandas as pd
import numpy as np
import sys
#sys.path.append('/Users/mengyanzhu/Documents/GitHub/msf-capstone-crm/')
#from Preprocess import preprocess_data
from sklearn.preprocessing import StandardScaler

def preprocess_NoWoE(file_years, labelled):
    #not sure how this works Preprocess.data_preparation(file_number)
    table= preprocess_data(file_years, labelled).dropna()
    if labelled:
        #table = pd.read_csv(file_name, low_memory=False).dropna()

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
        scaler = StandardScaler()
        X[num_col] = scaler.fit_transform(X[num_col])
        X = pd.concat([X[num_col], X[cat_col]], axis=1)
        data =pd.concat([X,y], axis = 1)

    else:
        table = table[[
            'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', # 'CSCORE_C', 
            'FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'NUM_UNIT', 
            'OCC_Principal', 'OCC_Second', 'OCC_Investor', 'MI_PCT', 'Ongoing', 
            'Current_DLQ', 'Prepaid_Matured']]
        num_col = ['ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B', # 'CSCORE_C', 
        'NUM_UNIT', 'MI_PCT']
        cat_col = ['FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'OCC_Principal', 'OCC_Second', 'OCC_Investor']
        scaler = StandardScaler()
        data = table.copy()
        data[num_col] = scaler.fit_transform(data[num_col])
        data = pd.concat([data[num_col], data[cat_col]], axis=1)
        
        
    
    return data






def preprocess_data(file_years, labelled):
    year_table = pd.DataFrame()
    path = "/Users/mengyanzhu/Documents/GitHub/msf-capstone-crm/dataset/"
    for file_year in file_years:
        for i in range (1, 5):
            
            file_name = path+ f"{file_year}Q{i}_stat.csv"
            print (file_name)
            if labelled:
                quarter_table = preprocess(pd.read_csv(file_name, low_memory=False))
            else:
                quarter_table = preprocess_nl(pd.read_csv(file_name, low_memory=False))
            
            if year_table.empty:
                year_table = quarter_table
            else:
                year_table = pd.concat([year_table, quarter_table], ignore_index=True)


    n_row, n_col = year_table.shape
    if labelled:
        n_dlq = sum(year_table['DLQ_FLAG'])
        print("this dataset is labelled")
        print(f"Delinquency percentage: {round(n_dlq / n_row * 100, 2)} %")
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
def data_preparation(table, file_name):
    
    table['ORIG_RATE'] = pd.to_numeric(table['ORIG_RATE'], errors='coerce')
    table['CURR_RATE'] = pd.to_numeric(table['CURR_RATE'], errors='coerce')
    
    lppub_base = table[[
        'LOAN_ID', 'ACT_PERIOD', 'CHANNEL', 'SELLER', 'SERVICER', 'ORIG_RATE', 'CURR_RATE',
        'ORIG_UPB', 'CURRENT_UPB', 'ORIG_TERM', 'ORIG_DATE', 'FIRST_PAY', 'LOAN_AGE', 'REM_MONTHS',
        'ADJ_REM_MONTHS', 'MATR_DT', 'OLTV', 'OCLTV', 'NUM_BO', 'DTI', 'CSCORE_B', 'CSCORE_C',
        'FIRST_FLAG', 'PURPOSE', 'PROP', 'NO_UNITS', 'OCC_STAT', 'STATE', 'MSA', 'ZIP', 'MI_PCT',
        'PRODUCT', 'DLQ_STATUS', 'MOD_FLAG', 'Zero_Bal_Code', 'ZB_DTE', 'LAST_PAID_INSTALLMENT_DATE',
        'FORECLOSURE_DATE', 'DISPOSITION_DATE', 'FORECLOSURE_COSTS', 'PROPERTY_PRESERVATION_AND_REPAIR_COSTS',
        'ASSET_RECOVERY_COSTS', 'MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS', 'ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY',
        'NET_SALES_PROCEEDS', 'CREDIT_ENHANCEMENT_PROCEEDS', 'REPURCHASES_MAKE_WHOLE_PROCEEDS', 'OTHER_FORECLOSURE_PROCEEDS',
        'NON_INTEREST_BEARING_UPB', 'PRINCIPAL_FORGIVENESS_AMOUNT', 'RELOCATION_MORTGAGE_INDICATOR', 'MI_TYPE',
        'SERV_IND', 'RPRCH_DTE', 'LAST_UPB'
    ]].copy()
    
    lppub_base.loc[:, 'repch_flag'] = np.where(lppub_base['RPRCH_DTE'].notna(), 1, 0)  # Whether the Repurchase Date is available
    
    lppub_base.loc[:, 'ACT_PERIOD'] = pd.to_datetime(lppub_base['ACT_PERIOD'].str[2:6] + '-' + lppub_base['ACT_PERIOD'].str[0:2] + '-01')
    lppub_base.loc[:, 'FIRST_PAY'] = pd.to_datetime(lppub_base['FIRST_PAY'].str[2:6] + '-' + lppub_base['FIRST_PAY'].str[0:2] + '-01')
    lppub_base.loc[:, 'ORIG_DATE'] = pd.to_datetime(lppub_base['ORIG_DATE'].str[2:6] + '-' + lppub_base['ORIG_DATE'].str[0:2] + '-01')
    lppub_base.loc[:, 'MATR_DT'] = pd.to_datetime(lppub_base['MATR_DT'].str[2:6] + '-' + lppub_base['MATR_DT'].str[0:2] + '-01')
    lppub_base.loc[:, 'ZB_DTE'] = pd.to_datetime(lppub_base['ZB_DTE'].str[2:6] + '-' + lppub_base['ZB_DTE'].str[0:2] + '-01')
    lppub_base.loc[:, 'LAST_PAID_INSTALLMENT_DATE'] = pd.to_datetime(lppub_base['LAST_PAID_INSTALLMENT_DATE'].str[2:6] + '-' + lppub_base['LAST_PAID_INSTALLMENT_DATE'].str[0:2] + '-01')
    lppub_base.loc[:, 'FORECLOSURE_DATE'] = pd.to_datetime(lppub_base['FORECLOSURE_DATE'].str[2:6] + '-' + lppub_base['FORECLOSURE_DATE'].str[0:2] + '-01')
    lppub_base.loc[:, 'DISPOSITION_DATE'] = pd.to_datetime(lppub_base['DISPOSITION_DATE'].str[2:6] + '-' + lppub_base['DISPOSITION_DATE'].str[0:2] + '-01')
    
    lppub_base = lppub_base.sort_values(by=['LOAN_ID', 'ACT_PERIOD'])
    
    del table

    
    acquisitionFile = lppub_base[[
        'LOAN_ID', 'ACT_PERIOD', 'CHANNEL', 'SELLER', 'ORIG_RATE', 'ORIG_UPB',
        'ORIG_TERM', 'ORIG_DATE', 'FIRST_PAY', 'OLTV', 'OCLTV', 'NUM_BO', 'DTI',
        'CSCORE_B', 'CSCORE_C', 'FIRST_FLAG', 'PURPOSE', 'PROP', 'NO_UNITS', 'OCC_STAT',
        'STATE', 'ZIP', 'MI_PCT', 'PRODUCT', 'MI_TYPE', 'RELOCATION_MORTGAGE_INDICATOR'
    ]].rename(columns={
        'CHANNEL': 'ORIG_CHN', 'ORIG_RATE': 'orig_rt', 'ORIG_UPB': 'orig_amt',
        'ORIG_TERM': 'orig_trm', 'ORIG_DATE': 'orig_date', 'FIRST_PAY': 'first_pay',
        'OLTV': 'oltv', 'OCLTV': 'ocltv', 'NUM_BO': 'num_bo', 'DTI': 'dti',
        'FIRST_FLAG': 'FTHB_FLG', 'PURPOSE': 'purpose', 'PROP': 'PROP_TYP',
        'NO_UNITS': 'NUM_UNIT', 'OCC_STAT': 'occ_stat', 'STATE': 'state', 'ZIP': 'zip_3',
        'MI_PCT': 'mi_pct', 'PRODUCT': 'prod_type', 'RELOCATION_MORTGAGE_INDICATOR': 'relo_flg'
    })

    # Summarize first period of acquisition data
    acqFirstPeriod = acquisitionFile.groupby('LOAN_ID').agg(first_period=('ACT_PERIOD', 'max')).reset_index()
    # Join the summarized data back to the original data
    acqFirstPeriod = acqFirstPeriod.merge(acquisitionFile, how='left', left_on=['LOAN_ID', 'first_period'], right_on=['LOAN_ID', 'ACT_PERIOD'])
    
    acqFirstPeriod = acqFirstPeriod[[
        'LOAN_ID', 'ORIG_CHN', 'SELLER', 'orig_rt', 'orig_amt', 'orig_trm', 'orig_date',
        'first_pay', 'oltv', 'ocltv', 'num_bo', 'dti', 'CSCORE_B', 'CSCORE_C', 'FTHB_FLG',
        'purpose', 'PROP_TYP', 'NUM_UNIT', 'occ_stat', 'state', 'zip_3', 'mi_pct', 'prod_type',
        'MI_TYPE', 'relo_flg'
    ]]
    
    acquisitionFile = acqFirstPeriod
    del acqFirstPeriod
    
    
    performanceFile = lppub_base[[
        'LOAN_ID', 'ACT_PERIOD', 'SERVICER', 'CURR_RATE', 'CURRENT_UPB', 'LOAN_AGE', 'REM_MONTHS', 'ADJ_REM_MONTHS', 
        'MATR_DT', 'MSA', 'DLQ_STATUS', 'MOD_FLAG', 'Zero_Bal_Code', 'ZB_DTE', 'LAST_PAID_INSTALLMENT_DATE', 
        'FORECLOSURE_DATE', 'DISPOSITION_DATE', 'FORECLOSURE_COSTS', 'PROPERTY_PRESERVATION_AND_REPAIR_COSTS', 
        'ASSET_RECOVERY_COSTS', 'MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS', 'ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY', 
        'NET_SALES_PROCEEDS', 'CREDIT_ENHANCEMENT_PROCEEDS', 'REPURCHASES_MAKE_WHOLE_PROCEEDS', 'OTHER_FORECLOSURE_PROCEEDS', 
        'NON_INTEREST_BEARING_UPB', 'PRINCIPAL_FORGIVENESS_AMOUNT', 'repch_flag', 'LAST_UPB'
    ]].rename(columns={
        'ACT_PERIOD': 'period', 'SERVICER': 'servicer', 'CURR_RATE': 'curr_rte', 'CURRENT_UPB': 'act_upb', 
        'LOAN_AGE': 'loan_age', 'REM_MONTHS': 'rem_mths', 'ADJ_REM_MONTHS': 'adj_rem_months', 'MATR_DT': 'maturity_date', 
        'MSA': 'msa', 'DLQ_STATUS': 'dlq_status', 'MOD_FLAG': 'mod_ind', 'Zero_Bal_Code': 'z_zb_code', 
        'ZB_DTE': 'zb_date', 'LAST_PAID_INSTALLMENT_DATE': 'lpi_dte', 'FORECLOSURE_DATE': 'fcc_dte', 
        'DISPOSITION_DATE': 'disp_dte', 'FORECLOSURE_COSTS': 'FCC_COST', 'PROPERTY_PRESERVATION_AND_REPAIR_COSTS': 'PP_COST', 
        'ASSET_RECOVERY_COSTS': 'AR_COST', 'MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS': 'IE_COST', 
        'ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY': 'TAX_COST', 'NET_SALES_PROCEEDS': 'NS_PROCS', 
        'CREDIT_ENHANCEMENT_PROCEEDS': 'CE_PROCS', 'REPURCHASES_MAKE_WHOLE_PROCEEDS': 'RMW_PROCS', 
        'OTHER_FORECLOSURE_PROCEEDS': 'O_PROCS', 'NON_INTEREST_BEARING_UPB': 'non_int_upb', 
        'PRINCIPAL_FORGIVENESS_AMOUNT': 'prin_forg_upb', 'LAST_UPB': 'zb_upb'
    })

    del lppub_base


    acquisition_year = file_name[:4]
    acquisition_qtr = file_name[4:6]
    
    performanceFile['servicer'] = performanceFile['servicer'].astype(str)
    performanceFile['z_zb_code'] = performanceFile['z_zb_code'].astype(str)
    
    if acquisition_qtr == 'Q1':
        acquisition_month = '03'
    elif acquisition_qtr == 'Q2':
        acquisition_month = '06'
    elif acquisition_qtr == 'Q3':
        acquisition_month = '09'
    else:
        acquisition_month = '12'
    
    acquisition_date = f"{acquisition_year}-{acquisition_month}-01"  # 2023-09-01
    
    # Convert all date fields to YYYY-MM-DD format
    acquisitionFile = acquisitionFile.rename(columns={
        'orig_date': 'ORIG_DTE',
        'first_pay': 'FRST_DTE'
    })
    # Old format: 7/1/2023  12:00:00 AM. New format: 7/1/2023
    acquisitionFile['ORIG_DTE'] = pd.to_datetime(acquisitionFile['ORIG_DTE'])
    acquisitionFile['FRST_DTE'] = pd.to_datetime(acquisitionFile['FRST_DTE'])
    performanceFile['period'] = pd.to_datetime(performanceFile['period'])
    performanceFile['maturity_date'] = pd.to_datetime(performanceFile['maturity_date'])
    performanceFile['zb_date'] = pd.to_datetime(performanceFile['zb_date'])
    performanceFile['lpi_dte'] = pd.to_datetime(performanceFile['lpi_dte'])
    performanceFile['fcc_dte'] = pd.to_datetime(performanceFile['fcc_dte'])
    performanceFile['disp_dte'] = pd.to_datetime(performanceFile['disp_dte'])


    baseTable1 = acquisitionFile.copy()
    
    baseTable1['AQSN_DTE'] = acquisition_date
    
    baseTable1['MI_TYPE'] = baseTable1['MI_TYPE'].replace({
        '1': 'BPMI',  # Borrower Paid Mortgage Insurance
        '2': 'LPMI',  # Lender Paid Mortgage Insurance
        '3': 'IPMI'   # Investor Paid Mortgage Insurance
    }).fillna('None')
    
    baseTable1['ocltv'] = np.where(baseTable1['ocltv'].isna(), baseTable1['oltv'], baseTable1['ocltv'])
    
    # LAST_ACTIVITY_DATE table: Get the latest activity date for each loan
    last_act_dte_table = performanceFile.groupby('LOAN_ID').agg(LAST_ACTIVITY_DATE=('period', 'max')).reset_index()
    
    # LAST_UPB table: Get the latest UPB for each loan, replacing with zb_upb if not missing
    last_upb_table = (performanceFile.groupby('LOAN_ID')
                      .apply(lambda x: x.loc[x['period'].idxmax()])
                      .assign(LAST_UPB=lambda x: np.where(pd.notna(x['zb_upb']), x['zb_upb'], x['act_upb']))
                      [['LOAN_ID', 'LAST_UPB']]
                      .reset_index(drop=True))
    
    # LAST_RT table: Get the latest interest rate for each loan
    last_rt_table = (performanceFile[performanceFile['curr_rte'].notna()]
                     .groupby('LOAN_ID').agg(LAST_RT_DATE=('period', 'max')).reset_index()
                     .merge(performanceFile, how='left', left_on=['LOAN_ID', 'LAST_RT_DATE'], right_on=['LOAN_ID', 'period'])
                     [['LOAN_ID', 'curr_rte']]
                     .rename(columns={'curr_rte': 'LAST_RT'})
                     .assign(LAST_RT=lambda x: x['LAST_RT'].round(3)))
    
    # zb_code_table: Get the zero-balance code for each loan
    zb_code_table = (performanceFile[performanceFile['z_zb_code'] != 'nan']
                     .groupby('LOAN_ID').agg(zb_code_dt=('period', 'max')).reset_index()
                     .merge(performanceFile, how='left', left_on=['LOAN_ID', 'zb_code_dt'], right_on=['LOAN_ID', 'period'])
                     [['LOAN_ID', 'z_zb_code']]
                     .rename(columns={'z_zb_code': 'zb_code'}))
    
    max_table = (last_act_dte_table
                 .merge(performanceFile, how='left', left_on=['LOAN_ID', 'LAST_ACTIVITY_DATE'], right_on=['LOAN_ID', 'period'])
                 .merge(last_upb_table, how='left', on='LOAN_ID')
                 .merge(last_rt_table, how='left', on='LOAN_ID')
                 .merge(zb_code_table, how='left', on='LOAN_ID'))
    
    del last_act_dte_table, last_upb_table, last_rt_table, zb_code_table
    
    # servicer_table: Get the latest servicer for each loan
    servicer_table = (performanceFile[performanceFile['servicer'] != 'nan']
                      .groupby('LOAN_ID').agg(servicer_period=('period', 'max')).reset_index()
                      .merge(performanceFile, how='left', left_on=['LOAN_ID', 'servicer_period'], right_on=['LOAN_ID', 'period'])
                      .assign(SERVICER=lambda x: x['servicer'])
                      [['LOAN_ID', 'SERVICER']])
    
    # non_int_upb_table: Get the second-to-last non-interest UPB for each loan
    non_int_upb_table = (performanceFile.groupby('LOAN_ID')
                         .apply(lambda x: x.iloc[-2] if len(x) > 1 else x.iloc[-1])
                         .reset_index(drop=True)[['LOAN_ID', 'non_int_upb']]
                         .rename(columns={'non_int_upb': 'NON_INT_UPB'})
                         .assign(NON_INT_UPB=lambda x: np.where(x['NON_INT_UPB'].isna(), 0, x['NON_INT_UPB'])))
    
    # Merge all the tables into baseTable2
    baseTable2 = (baseTable1
                  .merge(max_table, how='left', on='LOAN_ID')
                  .merge(servicer_table, how='left', on='LOAN_ID')
                  .merge(non_int_upb_table, how='left', on='LOAN_ID'))
    
    del max_table, servicer_table, non_int_upb_table


    fcc_table = performanceFile.dropna(subset=['lpi_dte', 'fcc_dte', 'disp_dte'])
    
    fcc_table = (fcc_table.groupby('LOAN_ID')
                 .agg(LPI_DTE=('lpi_dte', 'max'), FCC_DTE=('fcc_dte', 'max'), DISP_DTE=('disp_dte', 'max'))
                 .reset_index())
    
    baseTable3 = baseTable2.merge(fcc_table, how='left', on='LOAN_ID')
    
    del fcc_table, baseTable2


    # Create the series of "first DQ occurrence" tables and loan modification tables
    slimPerformanceFile = (performanceFile[['LOAN_ID', 'period', 'dlq_status', 'z_zb_code', 'act_upb', 'zb_upb', 'mod_ind', 'maturity_date', 'rem_mths']]
                           .copy())
    
    slimPerformanceFile['dlq_status'] = slimPerformanceFile['dlq_status'].replace('XX', '999').astype(int)
    
    f30_table = (slimPerformanceFile[(slimPerformanceFile['dlq_status'] >= 1) & (slimPerformanceFile['dlq_status'] < 999) & (slimPerformanceFile['z_zb_code'] == 'nan')]
                 .groupby('LOAN_ID').agg(F30_DTE=('period', 'min')).reset_index()
                 .merge(slimPerformanceFile, how='left', left_on=['LOAN_ID', 'F30_DTE'], right_on=['LOAN_ID', 'period'])
                 [['LOAN_ID', 'F30_DTE', 'act_upb']]
                 .rename(columns={'act_upb': 'F30_UPB'}))
    
    f60_table = (slimPerformanceFile[(slimPerformanceFile['dlq_status'] >= 2) & (slimPerformanceFile['dlq_status'] < 999) & (slimPerformanceFile['z_zb_code'] == 'nan')]
                 .groupby('LOAN_ID').agg(F60_DTE=('period', 'min')).reset_index()
                 .merge(slimPerformanceFile, how='left', left_on=['LOAN_ID', 'F60_DTE'], right_on=['LOAN_ID', 'period'])
                 [['LOAN_ID', 'F60_DTE', 'act_upb']]
                 .rename(columns={'act_upb': 'F60_UPB'}))
    
    f90_table = (slimPerformanceFile[(slimPerformanceFile['dlq_status'] >= 3) & (slimPerformanceFile['dlq_status'] < 999) & (slimPerformanceFile['z_zb_code'] == 'nan')]
                 .groupby('LOAN_ID').agg(F90_DTE=('period', 'min')).reset_index()
                 .merge(slimPerformanceFile, how='left', left_on=['LOAN_ID', 'F90_DTE'], right_on=['LOAN_ID', 'period'])
                 [['LOAN_ID', 'F90_DTE', 'act_upb']]
                 .rename(columns={'act_upb': 'F90_UPB'}))
    
    f120_table = (slimPerformanceFile[(slimPerformanceFile['dlq_status'] >= 4) & (slimPerformanceFile['dlq_status'] < 999) & (slimPerformanceFile['z_zb_code'] == 'nan')]
                  .groupby('LOAN_ID').agg(F120_DTE=('period', 'min')).reset_index()
                  .merge(slimPerformanceFile, how='left', left_on=['LOAN_ID', 'F120_DTE'], right_on=['LOAN_ID', 'period'])
                  [['LOAN_ID', 'F120_DTE', 'act_upb']]
                  .rename(columns={'act_upb': 'F120_UPB'}))
    
    f180_table = (slimPerformanceFile[(slimPerformanceFile['dlq_status'] >= 6) & (slimPerformanceFile['dlq_status'] < 999) & (slimPerformanceFile['z_zb_code'] == 'nan')]
                  .groupby('LOAN_ID').agg(F180_DTE=('period', 'min')).reset_index()
                  .merge(slimPerformanceFile, how='left', left_on=['LOAN_ID', 'F180_DTE'], right_on=['LOAN_ID', 'period'])
                  [['LOAN_ID', 'F180_DTE', 'act_upb']]
                  .rename(columns={'act_upb': 'F180_UPB'}))
    
    fce_table = (slimPerformanceFile[((slimPerformanceFile['z_zb_code'].isin(['02', '03', '09', '15'])) | (slimPerformanceFile['dlq_status'] >= 6)) & (slimPerformanceFile['dlq_status'] < 999)]
                 .groupby('LOAN_ID').agg(FCE_DTE=('period', 'min')).reset_index()
                 .merge(slimPerformanceFile, how='left', left_on=['LOAN_ID', 'FCE_DTE'], right_on=['LOAN_ID', 'period'])
                 [['LOAN_ID', 'FCE_DTE', 'act_upb', 'zb_upb']]
                 .assign(FCE_UPB=lambda x: x['zb_upb'] + x['act_upb'])
                 [['LOAN_ID', 'FCE_DTE', 'FCE_UPB']])


    fmod_dte_table = (slimPerformanceFile[(slimPerformanceFile['mod_ind'] == 'Y') & (slimPerformanceFile['z_zb_code'] == 'nan')]
                      .groupby('LOAN_ID').agg(FMOD_DTE=('period', 'min')).reset_index())
    
    slimPerformanceFile['period_year'] = slimPerformanceFile['period'].dt.year
    slimPerformanceFile['period_month'] = slimPerformanceFile['period'].dt.month
    
    fmod_dte_table['FMOD_DTE_year'] = fmod_dte_table['FMOD_DTE'].dt.year
    fmod_dte_table['FMOD_DTE_month'] = fmod_dte_table['FMOD_DTE'].dt.month
    
    fmod_table = (slimPerformanceFile[(slimPerformanceFile['mod_ind'] == 'Y') & (slimPerformanceFile['z_zb_code'] == 'nan')]
                  .merge(fmod_dte_table, how='left', on='LOAN_ID'))
    
    fmod_table['period_total_months'] = fmod_table['period_year'] * 12 + fmod_table['period_month']
    fmod_table['FMOD_DTE_total_months'] = fmod_table['FMOD_DTE_year'] * 12 + fmod_table['FMOD_DTE_month']
    
    fmod_table = fmod_table[fmod_table['period_total_months'] <= fmod_table['FMOD_DTE_total_months'] + 3]
    
    fmod_table = (fmod_table.groupby('LOAN_ID')
                  .agg(FMOD_UPB=('act_upb', 'max'))
                  .reset_index())
    
    fmod_table = (fmod_table.merge(fmod_dte_table, how='left', on='LOAN_ID')
                  .merge(slimPerformanceFile, how='left', left_on=['LOAN_ID', 'FMOD_DTE'], right_on=['LOAN_ID', 'period'])
                  [['LOAN_ID', 'FMOD_DTE', 'FMOD_UPB', 'maturity_date']])
    
    del fmod_dte_table
    
    f120_table['F120_DTE'] = f120_table['F120_DTE']
    acquisitionFile['FRST_DTE'] = acquisitionFile['FRST_DTE']
    
    num_120_table = (f120_table.merge(acquisitionFile, how='left', on='LOAN_ID')
                     .assign(z_num_periods_120=lambda x: (((x['F120_DTE'].dt.year * 12 + x['F120_DTE'].dt.month) -
                                                            (x['FRST_DTE'].dt.year * 12 + x['FRST_DTE'].dt.month) + 1)))
                     [['LOAN_ID', 'z_num_periods_120']])
    
    del acquisitionFile


    orig_maturity_table = (slimPerformanceFile.dropna(subset=['maturity_date'])
                           .groupby('LOAN_ID').agg(maturity_date_period=('period', 'min')).reset_index()
                           .merge(slimPerformanceFile, how='left', left_on=['LOAN_ID', 'maturity_date_period'], right_on=['LOAN_ID', 'period'])
                           [['LOAN_ID', 'maturity_date']]
                           .rename(columns={'maturity_date': 'orig_maturity_date'}))  # LOAN_ID here is a column label
    
    trm_chng_table = (slimPerformanceFile.groupby('LOAN_ID') 
                      .apply(lambda x: x.assign(prev_rem_mths=x['rem_mths'].shift(1))) 
                      .assign(trm_chng=lambda x: x['rem_mths'] - x['prev_rem_mths'], 
                              did_trm_chng=lambda x: np.where(x['trm_chng'].fillna(-1) >= 0, 1, 0)) 
                      .query('did_trm_chng == 1') 
                      .set_index("LOAN_ID") 
                      .groupby('LOAN_ID').agg(trm_chng_dt=('period', 'min')).reset_index()) 


    modtrm_table = (fmod_table.merge(orig_maturity_table, how='left', on='LOAN_ID')
                    .merge(trm_chng_table, how='left', on='LOAN_ID')
                    .assign(MODTRM_CHNG=lambda x: np.where((x['maturity_date'] != x['orig_maturity_date']) | x['trm_chng_dt'].notna(), 1, 0))
                    [['LOAN_ID', 'MODTRM_CHNG']])
    
    pre_mod_upb_table = (slimPerformanceFile.merge(fmod_table, how='left', on='LOAN_ID')
                         .query('period < FMOD_DTE')
                         .groupby('LOAN_ID').agg(pre_mod_period=('period', 'max')).reset_index()
                         .merge(slimPerformanceFile, how='left', left_on=['LOAN_ID', 'pre_mod_period'], right_on=['LOAN_ID', 'period'])
                         [['LOAN_ID', 'act_upb']]
                         .rename(columns={'act_upb': 'pre_mod_upb'}))
    
    modupb_table = (fmod_table.merge(pre_mod_upb_table, how='left', on='LOAN_ID')
                    .assign(MODUPB_CHNG=lambda x: np.where(x['FMOD_UPB'] >= x['pre_mod_upb'], 1, 0))
                    [['LOAN_ID', 'MODUPB_CHNG']])
    
    slimPerformanceFile['mod_ind'].unique()


    baseTable4 = (baseTable3
                  .merge(f30_table, how='left', on='LOAN_ID')
                  .merge(f60_table, how='left', on='LOAN_ID')
                  .merge(f90_table, how='left', on='LOAN_ID')
                  .merge(f120_table, how='left', on='LOAN_ID')
                  .merge(f180_table, how='left', on='LOAN_ID')
                  .merge(fce_table, how='left', on='LOAN_ID')
                  .merge(fmod_table, how='left', on='LOAN_ID')
                  .merge(num_120_table, how='left', on='LOAN_ID')
                  .merge(modtrm_table, how='left', on='LOAN_ID')
                  .merge(modupb_table, how='left', on='LOAN_ID')
                  .assign(
                      F30_UPB=lambda x: np.where(x['F30_UPB'].isna() & x['F30_DTE'].notna(), x['orig_amt'], x['F30_UPB']),
                      F60_UPB=lambda x: np.where(x['F60_UPB'].isna() & x['F60_DTE'].notna(), x['orig_amt'], x['F60_UPB']),
                      F90_UPB=lambda x: np.where(x['F90_UPB'].isna() & x['F90_DTE'].notna(), x['orig_amt'], x['F90_UPB']),
                      F120_UPB=lambda x: np.where(x['F120_UPB'].isna() & x['F120_DTE'].notna(), x['orig_amt'], x['F120_UPB']),
                      F180_UPB=lambda x: np.where(x['F180_UPB'].isna() & x['F180_DTE'].notna(), x['orig_amt'], x['F180_UPB']),
                      FCE_UPB=lambda x: np.where(x['FCE_UPB'].isna() & x['FCE_DTE'].notna(), x['orig_amt'], x['FCE_UPB'])
                  ))
    
    del baseTable3, f30_table, f60_table, f90_table, f120_table, f180_table, fce_table, fmod_table, num_120_table, modtrm_table, modupb_table
    del orig_maturity_table, trm_chng_table, slimPerformanceFile, pre_mod_upb_table
    

    # First assignment without PFG_COST
    baseTable4['disp_dte'] = baseTable4['disp_dte'].fillna(value=np.nan)
    baseTable4['disp_dte'] = pd.to_datetime(baseTable4['disp_dte'], errors='coerce')
    
    baseTable4['LPI_DTE'] = pd.to_datetime(baseTable4['LPI_DTE'], errors='coerce')
    
    baseTable5 = (baseTable4
                  .assign(
                      LAST_DTE=lambda x: np.where(~np.isnan(x['disp_dte']) , x['disp_dte'], x['LAST_ACTIVITY_DATE']),
                      repch_flag=lambda x: np.where(x['repch_flag'] == 'Y', 1, 0),
                      PFG_COST=lambda x: x['prin_forg_upb'],                           # PRINCIPAL_FORGIVENESS_AMOUNT
                      MOD_FLAG=lambda x: np.where(x['FMOD_DTE'].notna(), 1, 0),        # First modification date exists
                      MODFG_COST=lambda x: np.where(x['mod_ind'] == 'Y', 0, np.nan),   # Mortgage loan has been modified (?)
                      MODTRM_CHNG=lambda x: x['MODTRM_CHNG'].fillna(0),
                      MODUPB_CHNG=lambda x: x['MODUPB_CHNG'].fillna(0),
                      CSCORE_MN=lambda x: np.where(x['CSCORE_C'].notna() & x['CSCORE_B'].notna() & (x['CSCORE_C'] < x['CSCORE_B']), x['CSCORE_C'], x['CSCORE_B']),
                      ORIG_VAL=lambda x: round(x['orig_amt'] / (x['oltv'] / 100), 2),  # value of the property
                      dlq_status=lambda x: np.where(x['dlq_status'].isin(['X', 'XX']), '999', x['dlq_status']),
                      z_last_status=lambda x: x['dlq_status'].astype(float),           # The number of months the obligor is delinquent
                      LAST_STAT=lambda x: np.select(
                          [
                              x['zb_code'] == '09', x['zb_code'] == '03', x['zb_code'] == '02', 
                              x['zb_code'] == '06', x['zb_code'] == '15', x['zb_code'] == '16', 
                              x['zb_code'] == '01', (x['z_last_status'] < 999) & (x['z_last_status'] >= 9), 
                              x['z_last_status'] == 8, x['z_last_status'] == 7, x['z_last_status'] == 6,
                              x['z_last_status'] == 5, x['z_last_status'] == 4, x['z_last_status'] == 3,
                              x['z_last_status'] == 2, x['z_last_status'] == 1, x['z_last_status'] == 0
                          ],
                          ['F', 'S', 'T', 'R', 'N', 'L', 'P', '9', '8', '7', '6', '5', '4', '3', '2', '1', 'C'],
                          default='X'
                      ),
                      
                      FCC_DTE=lambda x: np.where(pd.isna(x['FCC_DTE']) & (x['LAST_STAT'].isin(['F', 'S', 'N', 'T'])), x['zb_date'], x['FCC_DTE']),
                      COMPLT_FLG=lambda x: np.where(~pd.isna(x['DISP_DTE']), 1, np.nan),
                      INT_COST=lambda x: np.where(
                          (x['COMPLT_FLG'] == 1) & (~pd.isna(x['LPI_DTE'])),
                          round(((x['LAST_DTE'].dt.year * 12 + x['LAST_DTE'].dt.month) -
                                 (x['LPI_DTE'].dt.year * 12 + x['LPI_DTE'].dt.month)) *
                                ((x['LAST_RT'] / 100) - 0.0035) / 12 * (x['LAST_UPB'] + (-1 * x['NON_INT_UPB'])), 2),
                          np.nan
                      ),
                      FCC_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['FCC_COST'].isna(), 0, x['FCC_COST']),
                      PP_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['PP_COST'].isna(), 0, x['PP_COST']),
                      AR_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['AR_COST'].isna(), 0, x['AR_COST']),
                      IE_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['IE_COST'].isna(), 0, x['IE_COST']),
                      TAX_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['TAX_COST'].isna(), 0, x['TAX_COST'])
                  ))
    
    baseTable5 = (baseTable5
                  .assign(INT_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['INT_COST'].isna(), 0, x['INT_COST'])))
    
    # Second assignment focusing on PFG_COST
    baseTable5 = (baseTable5
                  .assign(
                      CSCORE_MN=lambda x: np.where(x['CSCORE_MN'].isna(), x['CSCORE_B'], x['CSCORE_MN']),
                      MODFG_COST=lambda x: np.where((x['mod_ind'] == 'Y') & (x['PFG_COST'] > 0), x['PFG_COST'], 0),
                      PFG_COST=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['PFG_COST'].isna(), 0, x['PFG_COST']),
                      COMPLT_FLG=lambda x: np.where(~x['LAST_STAT'].isin(['F', 'S', 'N', 'T']), np.nan, x['COMPLT_FLG']),
                      CE_PROCS=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['CE_PROCS'].isna(), 0, x['CE_PROCS']),
                      NS_PROCS=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['NS_PROCS'].isna(), 0, x['NS_PROCS']),
                      RMW_PROCS=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['RMW_PROCS'].isna(), 0, x['RMW_PROCS']),
                      O_PROCS=lambda x: np.where((x['COMPLT_FLG'] == 1) & x['O_PROCS'].isna(), 0, x['O_PROCS']),
                      NET_LOSS=lambda x: np.round(np.where(
                          x['COMPLT_FLG'] == 1,
                          (x['LAST_UPB'] + x['FCC_COST'] + x['PP_COST'] + x['AR_COST'] + x['IE_COST'] +
                           x['TAX_COST'] + x['PFG_COST'] + x['INT_COST'] + -1 * x['NS_PROCS'] + -1 * x['CE_PROCS'] +
                           -1 * x['RMW_PROCS'] + -1 * x['O_PROCS']), np.nan), 2),
                      NET_SEV=lambda x: np.round(np.where(x['COMPLT_FLG'] == 1, x['NET_LOSS'] / x['LAST_UPB'], np.nan), 6)
                  ))
    
    baseTable5 = (baseTable5
                  .assign(CSCORE_MN=lambda x: np.where(x['CSCORE_MN'].isna(), x['CSCORE_C'], x['CSCORE_MN'])))

    del baseTable4


    modir_table = (baseTable1
                   .merge(performanceFile, on='LOAN_ID', how='left')
                   .query("mod_ind == 'Y'")
                   .assign(
                       non_int_upb=lambda x: np.where(x['non_int_upb'].isna(), 0, x['non_int_upb']),
                       modir_cost=lambda x: np.round(np.where(x['mod_ind'] == 'Y', ((x['orig_rt'] - x['curr_rte']) / 1200) * x['act_upb'], 0), 2),
                       modfb_cost=lambda x: np.round(np.where((x['mod_ind'] == 'Y') & (x['non_int_upb'] > 0), (x['curr_rte'] / 1200) * x['non_int_upb'], 0), 2)
                   )
                   .groupby('LOAN_ID', as_index=False)
                   .agg(
                       MODIR_COST=('modir_cost', 'sum'),
                       MODFB_COST=('modfb_cost', 'sum')
                   )
                   .assign(
                       MODTOT_COST=lambda x: round(x['MODFB_COST'] + x['MODIR_COST'], 2)
                   ))
    
    del performanceFile


    baseTable5['zb_date'] = pd.to_datetime(baseTable5['zb_date'], errors='coerce')
    
    baseTable6 = (baseTable5
                  .merge(modir_table, on='LOAN_ID', how='left')
                  .assign(
                      COMPLT_FLG=lambda x: x['COMPLT_FLG'].astype(str)
                  ))
    
    baseTable6 = (baseTable6
                  .merge(modir_table, on='LOAN_ID', how='left')
                  .assign(
                      COMPLT_FLG=lambda x: np.where(pd.isna(x['COMPLT_FLG']), '', x['COMPLT_FLG'])
                  ))
    
    baseTable6 = (baseTable6
                  .merge(modir_table, on='LOAN_ID', how='left')
                  .assign(
                      non_int_upb=lambda x: np.where((x['COMPLT_FLG'] == '1') & (pd.isna(x['non_int_upb'])), 0, x['non_int_upb']),
                      MODIR_COST=lambda x: np.round(np.where(
                          x['COMPLT_FLG'] == '1', 
                          x['MODIR_COST'] +  (((x['LAST_DTE'].dt.year * 12 + x['LAST_DTE'].dt.month) - (x['zb_date'].dt.year * 12 + x['zb_date'].dt.month)) *
                                             (x['orig_rt'] - x['LAST_RT']) / 1200) * x['LAST_UPB'], 
                          x['MODIR_COST']
                      ), 2),
                      MODFB_COST=lambda x: np.round(np.where(
                          x['COMPLT_FLG'] == '1',
                          x['MODFB_COST'] + (((x['LAST_DTE'].dt.year * 12 + x['LAST_DTE'].dt.month) - (x['zb_date'].dt.year * 12 + x['zb_date'].dt.month)) * 
                                             (x['LAST_RT'])/ 1200) * x['non_int_upb'],
                          x['MODFB_COST']
                      ), 2),
                      COMPLT_FLG=lambda x: x['COMPLT_FLG'].astype(float),
                      orig_rt=lambda x: np.round(x['orig_rt'].astype(float), 3)
                  ))

    del baseTable1, baseTable5, modir_table


    baseTable7 = baseTable6[[
        'LOAN_ID', 'ORIG_CHN', 'SELLER', 'orig_rt', 'orig_amt', 'orig_trm', 'oltv', 'ocltv', 'num_bo', 'dti',
        'CSCORE_B', 'FTHB_FLG', 'purpose', 'PROP_TYP', 'NUM_UNIT', 'occ_stat', 'state', 'zip_3', 'mi_pct', 'CSCORE_C',
        'relo_flg', 'MI_TYPE', 'AQSN_DTE', 'ORIG_DTE', 'FRST_DTE', 'LAST_RT', 'LAST_UPB', 'msa', 'FCC_COST', 'PP_COST',
        'AR_COST', 'IE_COST', 'TAX_COST', 'NS_PROCS', 'CE_PROCS', 'RMW_PROCS', 'O_PROCS', 'repch_flag', 'LAST_ACTIVITY_DATE',
        'LPI_DTE', 'FCC_DTE', 'DISP_DTE', 'SERVICER', 'F30_DTE', 'F60_DTE', 'F90_DTE', 'F120_DTE', 'F180_DTE', 'FCE_DTE',
        'F180_UPB', 'FCE_UPB', 'F30_UPB', 'F60_UPB', 'F90_UPB', 'MOD_FLAG', 'FMOD_DTE', 'FMOD_UPB', 'MODIR_COST', 'MODFB_COST',
        'MODFG_COST', 'MODTRM_CHNG', 'MODUPB_CHNG', 'z_num_periods_120', 'F120_UPB', 'CSCORE_MN', 'ORIG_VAL', 'LAST_DTE',
        'LAST_STAT', 'COMPLT_FLG', 'INT_COST', 'PFG_COST', 'NET_LOSS', 'NET_SEV', 'MODTOT_COST'
    ]]
    
    del baseTable6


    baseTable7['CSCORE_MN'] = baseTable7['CSCORE_MN'].astype('Int64')

    return baseTable7




def preprocess(table):
    
    table = table[[
        'LOAN_ID', 'orig_rt', 'orig_amt', 'orig_trm', 'oltv', 'num_bo', 'dti',
        'CSCORE_B', 'FTHB_FLG', 'purpose', 'PROP_TYP', 'NUM_UNIT', 'occ_stat',
        'mi_pct', 'CSCORE_C', 'MI_TYPE', 'AQSN_DTE', 'ORIG_DTE', 'FRST_DTE',
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
    
    table['AQSN_DTE'] = pd.to_datetime(table['AQSN_DTE'])
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

    table['DLQ_FLAG'] = table[['F30_DTE', 'F60_DTE', 'F90_DTE', 'F120_DTE', 'F180_DTE', 'FCE_DTE']].notna().any(axis=1).astype(int)

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
        'MI_PCT', 'MI_Borrower', 'MI_Lender', 'MI_Investor', 'AQSN_DTE', 'ORIG_DTE', 'FRST_DTE', 
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
        'mi_pct', 'CSCORE_C', 'MI_TYPE', 'AQSN_DTE', 'ORIG_DTE', 'FRST_DTE',
        'LAST_RT', 'LAST_UPB', 'LAST_ACTIVITY_DATE', 'ORIG_VAL','LAST_STAT'
        
    ]].rename(columns={
        'orig_rt': 'ORIG_RATE', 'orig_amt': 'ORIG_AMOUNT', 'orig_trm': 'ORIG_TERM',
        'oltv': 'OLTV', 'num_bo': 'NUM_BO', 'dti': 'DTI', 'FTHB_FLG': 'FTHB_FLAG',
        'purpose': 'PURPOSE', 'PROP_TYP': 'PROP_TYPE', 'occ_stat': 'OCC_STAT', 
        'mi_pct': 'MI_PCT', 'LAST_RT': 'LAST_RATE', 'ORIG_VAL': 'PROP_VALUE'
    })
    
    table['AQSN_DTE'] = pd.to_datetime(table['AQSN_DTE'])
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
        'MI_PCT', 'MI_Borrower', 'MI_Lender', 'MI_Investor', 'AQSN_DTE', 'ORIG_DTE', 'FRST_DTE', 
        'LAST_RATE', 'LAST_UPB', 'LAST_ACTIVITY_DATE', 'Ongoing', 'Current_DLQ', 'Prepaid_Matured', 'Third_Party_Sale', 
        'Short_Sale', 'Repurchased', 'Deed_In_Lieu', 'Non_Performing_NS', 'Reperforming_NS'
    ]]
    
    del table
    # print("The total number of NA is ", sum(newTable.isna().sum()))

    return newTable
    



def data_preparation(file_number):
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