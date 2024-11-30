import pandas as pd
import numpy as np
import os
import sys, os, glob
sys.path.append('/Users/mengyanzhu/Documents/GitHub/msf-capstone-crm/loan_data')
import prepare_func

def yearly_data():

    file_years = ['2022','2023']
    path = "../dataset/"
    year_table = pd.DataFrame()
    for file_year in file_years:
        for i in range (1, 5):
            
            file_name = path+ f"{file_year}Q{i}_stat.csv"
            print (file_name)
            quarter_table = prepare_func.preprocess(pd.read_csv(file_name, low_memory=False), 0)
            
            if year_table.empty:
                year_table = quarter_table
            else:
                year_table = pd.concat([year_table, quarter_table], ignore_index=True)


        n_row, n_col = year_table.shape
        #n_dlq = sum(year_table['DLQ_FLAG'])
        
    return year_table


