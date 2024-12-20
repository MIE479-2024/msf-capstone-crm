import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings


def preprocess_NoWoE(data: pd.DataFrame, labelled: bool) -> pd.DataFrame:
    """
    Preprocess the input data without Weight of Evidence (WoE) encoding.

    Args:
        data (pd.DataFrame): The input loan dataset.
        labelled (bool): Indicates if the dataset includes labels for evaluation.

    Returns:
        pd.DataFrame: The preprocessed dataset with scaled features.

    Raises:
        UserWarning: Warning regarding filtering based on DLQ_ and Ongoing flag.
    """
    data = filter_columns(data).astype(float)
    orig_length = len(data)
    if labelled:
        data = data[ (data['DLQ_90_FLAG'] == 1) | ( (data['DLQ_90_FLAG'] == 0) & (data['Ongoing'] == 1) ) ]
        warnings.warn("\nWhen testing on labelled data, dataset is filtered " +
                      "to include loans which match either of the following conditions: \n" +
                      "\t (i) Loan is 90 day deliquent or \n \t(ii) Loan is not 90 day deliquent " +
                      "but Ongoing.\nOut of the provided dataset " + str(len(data)) + " out of " + 
                      str(orig_length) + " loans match this definition.")

    num_col = [
        'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B',
        'NUM_UNIT', 'MI_PCT'
    ]
    cat_col = [
        'FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 'OCC_Principal',
        'OCC_Second', 'OCC_Investor'
    ]
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(data[num_col])
    output = pd.DataFrame(scaled_numeric, columns=num_col, index=data.index)
    output = pd.concat([output, data[cat_col]], axis=1)
    output['DLQ_90_FLAG'] = data["DLQ_90_FLAG"]
    return output


def filter_columns(table: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and preprocess columns in the loan dataset.

    Args:
        table (pd.DataFrame): The raw input loan dataset.

    Returns:
        pd.DataFrame: The filtered and preprocessed dataset.

    Raises:
        ValueError: If any required column in the dataset is completely empty.
    """
    for column in [
        'orig_rt', 'orig_amt', 'orig_trm', 'oltv', 'num_bo', 'dti',
        'CSCORE_B', 'FTHB_FLG', 'purpose', 'NUM_UNIT', 'occ_stat', 'mi_pct'
    ]:
        if table[column].isna().all():
            raise ValueError(f"The column '{column}' is completely empty, but required for classification models. Please provide valid data.")

    
    table = table[[
        'orig_rt', 'orig_amt', 'orig_trm', 'oltv', 'num_bo', 'dti',
        'CSCORE_B', 'FTHB_FLG', 'purpose', 'NUM_UNIT', 'occ_stat', 'mi_pct',
        'F30_DTE', 'F90_DTE', 'LAST_STAT'
    ]].rename(columns={
        'orig_rt': 'ORIG_RATE', 'orig_amt': 'ORIG_AMOUNT', 'orig_trm': 'ORIG_TERM',
        'oltv': 'OLTV', 'num_bo': 'NUM_BO', 'dti': 'DTI', 
        'FTHB_FLG': 'FTHB_FLAG', 'purpose': 'PURPOSE', 
        'occ_stat': 'OCC_STAT', 'mi_pct': 'MI_PCT'
    })

    # Handle missing values
    table['ORIG_RATE'] = table['ORIG_RATE'].fillna(table['ORIG_RATE'].median())
    table['DTI'] = table['DTI'].fillna(table['DTI'].median())
    table['CSCORE_B'] = table['CSCORE_B'].fillna(table['CSCORE_B'].median())
    table['MI_PCT'] = table['MI_PCT'].fillna(0)
    table['FTHB_FLAG'] = table['FTHB_FLAG'].replace({'Y': 1, 'N': 0})

    # Create one-hot encoding columns for categorical features
    table['PUR_Cash_out'] = (table['PURPOSE'] == 'C').astype(int)
    table['PUR_Refinance'] = (table['PURPOSE'] == 'R').astype(int)
    table['PUR_Purchase'] = (table['PURPOSE'] == 'P').astype(int)
    table['OCC_Principal'] = (table['OCC_STAT'] == 'P').astype(int)
    table['OCC_Second'] = (table['OCC_STAT'] == 'S').astype(int)
    table['OCC_Investor'] = (table['OCC_STAT'] == 'I').astype(int)

    # Create flags for delinquency and status
    table['DLQ_90_FLAG'] = table['F90_DTE'].notna().astype(int)
    table['Ongoing'] = (table['LAST_STAT'] == 'C').astype(int)

    # Select only required columns for output
    table = table[[
        'ORIG_RATE', 'ORIG_AMOUNT', 'ORIG_TERM', 'OLTV', 'NUM_BO', 'DTI', 'CSCORE_B',
        'FTHB_FLAG', 'PUR_Cash_out', 'PUR_Refinance', 'PUR_Purchase', 
        'NUM_UNIT', 'OCC_Principal', 'OCC_Second', 'OCC_Investor', 
        'MI_PCT', 'DLQ_90_FLAG', 'Ongoing'
    ]]
    return table
