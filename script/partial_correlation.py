"""
Partial correlation measures the degree of association between two random variables, x and y,
with the effect of a set of controlling random variables z removed.
There is a `pandas.DataFrame.corr`, but no built-in partial correlation support is available.
Usage: given a pandas DataFrame `df`, simply call `par_corr(df)`.
"""

import numpy as np
import pandas as pd
def par_corr_between(x, y, z):
    """
    Partial correlation coefficients between two variables x and y with respect to
    the control variable z.
    @param x (m, ) 1d array
    @param y (m, ) 1d array
    @param z (m, ) or (m, k), 1d or 2d array. If a 2d array, then each column corresponds to a variable.
	@return float, the partial correlation coefficient
    """
    assert x.ndim == 1
    assert y.ndim == 1
    assert z.ndim == 1 or z.ndim == 2
    if z.ndim == 1:
        z = np.reshape(z, (-1, 1))
    # solve two linear regression problems Zw = x and Zw = y
    Z = np.hstack([z, np.ones((z.shape[0], 1))])  # bias
    wx = np.linalg.lstsq(Z, x, rcond=None)[0]
    rx = x - Z @ wx # residual
    wy = np.linalg.lstsq(Z, y, rcond=None)[0]
    ry = y - Z @ wy
    # compute the Pearson correlation coefficient between the two residuals
    return np.corrcoef(rx, ry)[0, 1]
	
	
def par_corr(data_df):
    """
    Compute partial pairwise correlation of columns. 
    When a pair of columns are picked, then all other columns are treated as control variables. 
    
    @param data_df DataFrame
    @return DataFrame, whose data is a symmetric matrix
    """ 
    n = data_df.shape[1]
    mat = np.empty((n, n))
    np.fill_diagonal(mat, 1)
    for i in range(n):
        for j in range(i + 1, n):
            x = data_df.iloc[:, i].values
            y = data_df.iloc[:, j].values
            z = data_df.iloc[:, [t for t in range(data_df.shape[1]) if t != i and t != j]].values
            corr = par_corr_between(x, y, z)
            mat[i, j] = corr
            mat[j, i] = corr
    return pd.DataFrame(mat, index=data_df.columns, columns=data_df.columns)