import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt

def visual_patial_correlation(df_all:pd.DataFrame, target_variable:str, verbose:bool = False)->None:
    """
    Visualize the partial correlation coefficients between a target variable and all other variables in a dataframe.

    Parameters
    ----------
    df_all : pd.DataFrame
        The dataframe containing the time series data.
    target_variable : str
        The column name of the variable for which to compute the partial correlations.
    verbose : bool, optional
        Whether to print the partial correlations during computation. Defaults to False.

    Returns
    -------
    None
    """

    # The other variables to correlate with 'EUA'
    other_variables = [col for col in df_all.columns if col != target_variable]

    # Dictionary to store the partial correlation results
    partial_correlations = {}
    if verbose:
        print("Calculating partial correlations...")

    # Iterate through each variable to compute its partial correlation with 'EUA'
    for var in other_variables:
        # The control variables are all other variables except 'EUA' and the current 'var'
        control_variables = [col for col in other_variables if col != var]

        # Calculate the partial correlation
        # We select only the necessary columns for the calculation
        pcorr_result = pg.partial_corr(data=df_all, x=target_variable, y=var, covar=control_variables)

        # Store the correlation coefficient 'r'
        partial_correlations[var] = pcorr_result['r'].iloc[0]
        if verbose: print(f"  - Partial correlation between '{target_variable}' and '{var}': {pcorr_result['r'].iloc[0]:.4f}")


    # Convert the results to a pandas Series for easier plotting
    pcorr_series = pd.Series(partial_correlations)

    # --- 3. Visualize the Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create the bar plot
    sns.barplot(x=pcorr_series.index, y=pcorr_series.values, palette="viridis", ax=ax)

    # Add titles and labels
    ax.set_title(f'Partial Correlation of other variables with {target_variable}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Variables', fontsize=12)
    ax.set_ylabel('Partial Correlation Coefficient (r)', fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add the value on top of each bar
    for index, value in enumerate(pcorr_series.values):
        ax.text(index, value + (0.01 if value > 0 else -0.02), f'{value:.3f}',
                ha='center', va='bottom' if value > 0 else 'top', fontsize=10)

    # Ensure everything fits
    plt.tight_layout()

    # Show the plot
    plt.show()