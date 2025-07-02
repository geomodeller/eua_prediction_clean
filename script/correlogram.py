import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visual_all_cross_correlogram(df_all:pd.DataFrame,
                                 col1:str,
                                 col2_lst:list[str],
                                 total_lag:int,
                                 vline = None):
    plt.figure(figsize = (10, 3*len(col2_lst)))
    for j, col2 in enumerate(col2_lst):
        corss_corr = []
        for lag in range(1,total_lag):
            tail = []
            head = []
            for i in range(len(df_all)-lag):
                head.append(df_all.loc[i    , col1])
                tail.append(df_all.loc[i+lag, col2])
            corss_corr.append(np.corrcoef(head, tail)[0][1])
        plt.subplot(len(col2_lst), 1, j+1)
        plt.plot(range(1,total_lag), corss_corr)
        plt.title(f'{col1} vs. {col2}')
        plt.grid('on')
        if vline is not None:
            plt.axvline(x = vline, color = 'red')

def compute_cross_correlation(df_all:pd.DataFrame,
                              col1:str,
                              col2:str,
                              total_lag: int = 60,
                              visual_output: bool = True, 
                              return_corr: bool = True)->np.ndarray:
    """_summary_

    Args:
        df_all (pd.DataFrame): _description_ ff
        col1 (str): _description_
        col2 (str): _description_
        total_lag (int, optional): _description_. Defaults to 60.
        visual_output (bool, optional): _description_. Defaults to True.
        return_corr (bool, optional): _description_. Defaults to True.

    Returns:
        np.ndarray: _description_
    """
    col1 = 'EUA'
    col2 = 'Oil'
    corss_corr = []
    for lag in range(1,total_lag):
        tail = []
        head = []
        for i in range(len(df_all)-lag):
            head.append(df_all.loc[i    , col1])
            tail.append(df_all.loc[i+lag, col2])
        corss_corr.append(np.corrcoef(head, tail)[0][1])
    if visual_output:
        plt.plot(range(1,total_lag), corss_corr)
        plt.grid('on')
    if return_corr:
        return np.array(corss_corr)
    
def compute_auto_correlation(df_all: pd.DataFrame, 
                             col: str, 
                             total_lag: int = 60, 
                             visual_output: bool = True, 
                             return_corr: bool = True)->np.ndarray:
    """compute auto correlation of variable

    Args:
        df_all (pd.DataFrame): Data frame of time seriese info. 
        col (str): column name
        total_lag (int, optional): max lag. Defaults to 60.
        visual_output (bool, optional): whether to visual output. Defaults to True.
        return_corr (bool, optional): whether to return output. Defaults to True.

    Returns:
        np.ndarray: auto correlation
    """
    auto_corr = []
    for lag in range(1,total_lag):
        tail = []
        head = []
        for i in range(len(df_all)-lag):
            head.append(df_all.loc[i    , col])
            tail.append(df_all.loc[i+lag, col])
        auto_corr.append(np.corrcoef(head, tail)[0][1])

    if visual_output:
        plt.plot(range(1,total_lag), auto_corr)
        plt.grid('on')
    if return_corr:
        return np.array(auto_corr)



def plot_cross_correlogram(data1, data2, lags=20, title = None):
    """
    Plot the cross correlogram for two time series.
    
    Parameters:
    - data1: A pandas Series of the first time series data.
    - data2: A pandas Series of the second time series data.
    - lags: Number of lags to compute.
    """
    # Calculate the cross-correlation
    cross_corr = [data1.corr(data2.shift(lag)) for lag in range(-lags, lags + 1)]

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(-lags, lags + 1), cross_corr, color='green', alpha=0.6)
    if title == None:
        plt.title('Cross Correlogram')
    else:
        plt.title(title)
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axhline(1.96/np.sqrt(len(data1)), color='red', linestyle='--', label='95% CI')
    plt.axhline(-1.96/np.sqrt(len(data1)), color='red', linestyle='--')
    # plt.xticks(range(-lags, lags + 1))
    plt.legend()
    plt.grid()
    plt.show()
    

def plot_autocorrelogram(data, lags=20):
    """
    Plot the autocorrelogram for the given time series data.
    
    Parameters:
    - data: A pandas Series of time series data.
    - lags: Number of lags to compute.
    """
    # Calculate the autocorrelation
    autocorr = [data.autocorr(lag) for lag in range(lags + 1)]

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(lags + 1), autocorr, color='blue', alpha=0.6)
    plt.title('Autocorrelogram')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    # plt.xticks(range(lags + 1))
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axhline(1.96/np.sqrt(len(data)), color='red', linestyle='--', label='95% CI')
    plt.axhline(-1.96/np.sqrt(len(data)), color='red', linestyle='--')
    plt.legend()
    plt.grid()
    plt.show()