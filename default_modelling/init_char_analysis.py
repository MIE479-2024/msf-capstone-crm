"""Useful functions for perfoming Initial Characteristic Analysis"""

import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_distribution(
        data: pd.DataFrame,
        bin: pd.DataFrame,
        target: str = 'bad'
) -> float:
    """Calculates the distribution of "bad", "good" or "unknown" loans.

    Args:
        data: pd.DataFrame
            Entire dataset of loans being considered
        bin: pd.DataFrame
            Subset of loans 
        target: str
            Indicates which category of loans is being counted.

    Returns:
        float
    """
    ####  TODO: fix the logic here #### 

    if target == "bad":
        return bin["DLQ_FLAG"].sum() / data["DLQ_FLAG"].sum()
    
    elif target == "good":
        return (len(bin) - bin["DLQ_FLAG"].sum()) / (len(data)-(data["DLQ_FLAG"].sum()))
    # elif target == "unknown":
    
    ...


def get_measures(
        orig_data: pd.DataFrame,
        grouped_data: pd.api.typing.DataFrameGroupBy,
        plot: bool = True
) -> pd.DataFrame:
    """Calculates the WOE and IV of a set of loans grouped by a predefined binning.

    Args:
        data: GroupBy object
            Loan data grouped according to preselected bins.
        dlg_flag: str
            Column in data used to identify instances of "Bad".
        plot: bool
            Whether to call plot_woe_trend()
    
    Returns:
        pd.DataFrame
            Binnings and associated WOE and IV values. 
        float
            Total characteristic IV.
    """
    results = []
    for bin, group in grouped_data:
        bad_distribution = calculate_distribution(orig_data, group, target="bad")
        good_distribution = calculate_distribution(orig_data, group, target="good")
        if len(group) < 0.05*len(orig_data):
            print("WARNING: less than 5% of data in group", bin)
        if bad_distribution > 0 and good_distribution > 0:
            woe_measure = math.log(good_distribution/bad_distribution) * 100
            iv_measure = (
                (good_distribution - bad_distribution) * math.log(good_distribution/bad_distribution)
            )
        else:
            print("Bin either does not contain any 'bad' or 'good' observations.")
            woe_measure = np.nan
            iv_measure = np.nan

        results.append({
            'bin': str(bin),
            'woe': woe_measure,
            'iv': iv_measure
        })
    results = pd.DataFrame(results)
    results.set_index("bin", inplace=True)
    if plot:
        plot_trend(results, measure="woe")
        plot_trend(results, measure="iv")
    
    return results, results["iv"].sum()
    



def plot_trend(binned_df: pd.DataFrame, measure: str) -> plt.axis:
    """Creates a plot of WOE or IV using input dataframe.

    Args: 
        binned_df: DataFrame
            Binned data. i.e. each row is of the form 
            [index = bin, WOE (float)]
    Returns
        ax
    """
    plt.close()
    plt.figure(figsize=(max(10, len(binned_df) * 2), 6)) 

    sns.set_theme(style="whitegrid", palette="deep")
    ax = sns.barplot(
        data=binned_df,
        x=binned_df.index,
        y=measure,
    )
    ax.set_title("Predictive strength: " + measure)
    plt.plot(binned_df[measure], color='black', marker='o', linestyle='-')
    plt.show()
    return ax