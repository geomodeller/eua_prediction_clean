
import pandas as pd
from sklearn.preprocessing import StandardScaler
from script.create_sequences import create_sequences, create_sequences_many

def curate_training_test_data(df_all: pd.DataFrame, 
                              Date_col_name: str = 'Date',
                              sequence_length: int = 7, 
                              test_date: str | pd.Timestamp = '2024-07-01', 
                              flatten: bool = False,
                              predictors_lst: list[str] = ['EUA', 'Oil', 'Coal', 'NG', 'USEU', 'S&P_clean', 'DAX']):

    """
    Curate the training and test datasets from a given dataframe with time series data.

    This function processes the data to prepare training and test datasets for time series analysis
    by scaling the data and generating sequences of specified length.

    Parameters
    ----------
    df_all : pd.DataFrame
        The dataframe containing the time series data.
    Date_col_name : str, optional
        The column name of the date in the dataframe. Defaults to 'Date'.
    sequence_length : int, optional
        The length of the sequences to generate. Defaults to 7.
    test_date : str or pd.Timestamp, optional
        The starting date for the test data. Defaults to '2024-07-01'.
    flatten : bool, optional
        Indicates whether to flatten the data sequences (for non-LSTM models). Defaults to False.
    predictors_lst : list of str, optional
        A list of column names to use as predictors. Defaults to 
        ['EUA', 'Oil', 'Coal', 'NG', 'USEU', 'S&P_clean', 'DAX'].

    Returns
    -------
    tuple
        A tuple containing:
        - X_train : np.ndarray
            The training data sequences.
        - y_train : np.ndarray
            The training labels.
        - X_test : np.ndarray
            The test data sequences.
        - y_test : np.ndarray
            The test labels.
        - scaler : StandardScaler
            The scaler object fitted to the training data.
    """
    test_date = pd.to_datetime(test_date)

    test_overlap_time = pd.to_timedelta(sequence_length+1, unit = 'day')

    df_train = df_all[df_all[Date_col_name] < test_date].reset_index(drop=True)
    df_test  = df_all[df_all[Date_col_name] > test_date - test_overlap_time].reset_index(drop=True)

    train_data = df_train[predictors_lst].values 
    test_data  = df_test[predictors_lst].values
    scaler = StandardScaler()
    scaler.fit(train_data)

    train_data_scaled = scaler.transform(train_data)
    test_data_scaled  = scaler.transform(test_data)
    X_train, y_train = create_sequences(train_data_scaled, sequence_length, flatten = flatten) # LSTM should be flatten = False
    X_test, y_test = create_sequences(test_data_scaled, sequence_length, flatten = flatten)
    return X_train, y_train, X_test, y_test, scaler


def curate_training_test_data_many(df_all, Date_col_name = 'Date',
                                    sequence_length_input = 7,
                                    sequence_length_out = 7, 
                                    test_date= '2024-07-01', 
                                    flatten = False,
                                    predictors_lst = ['EUA', 'Oil', 'Coal', 'NG', 'USEU', 'S&P_clean', 'DAX'],
                                    dtype = 'float32',
                                    is_test_split = True):
    """
    Curate the training and test data, given the dataframe containing the time series data (df_all),
    the column name of the date in the dataframe (Date_col_name), the sequence length for the input data (sequence_length_input),
    the sequence length for the output data (sequence_length_out), the test data start date (test_date),
    whether to flatten the data or not (flatten), and the list of predictors to use (predictors_lst).

    Parameters
    ----------
    df_all : pandas.DataFrame
        The dataframe containing the time series data.
    Date_col_name : str, optional
        The column name of the date in the dataframe. Defaults to 'Date'.
    sequence_length_input : int, optional
        The sequence length for the input data. Defaults to 7.
    sequence_length_out : int, optional
        The sequence length for the output data. Defaults to 7.
    test_date : str or datetime, optional
        The test data start date. Defaults to '2024-07-01'.
    flatten : bool, optional
        Whether to flatten the data or not. Defaults to False.
    predictors_lst : list, optional
        The list of predictors to use. Defaults to ['EUA', 'Oil', 'Coal', 'NG', 'USEU', 'S&P_clean', 'DAX'].

    Returns
    -------
    X_train, y_train, X_test, y_test, scaler : tuple
        The training data, training labels, test data, test labels, and the scaler object.
    """
    test_date = pd.to_datetime(test_date)

    test_overlap_time = pd.to_timedelta(sequence_length_input+1, unit = 'day')
    if is_test_split:
        df_train = df_all[df_all[Date_col_name] < test_date].reset_index(drop=True)
        df_test  = df_all[df_all[Date_col_name] > test_date - test_overlap_time].reset_index(drop=True)

        train_data = df_train[predictors_lst].values 
        test_data  = df_test[predictors_lst].values
        scaler = StandardScaler()
        scaler.fit(train_data)

        train_data_scaled = scaler.transform(train_data)
        test_data_scaled  = scaler.transform(test_data)
        X_train, y_train = create_sequences_many(train_data_scaled, sequence_length_input, sequence_length_out, flatten = flatten,
                                                dtype = dtype) # LSTM should be flatten = False
        X_test, y_test = create_sequences_many(test_data_scaled, sequence_length_input, sequence_length_out, flatten = flatten,
                                            dtype= dtype)
        
    else:
        df_train = df_all

        train_data = df_train[predictors_lst].values 
        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data_scaled = scaler.transform(train_data)
        X_train, y_train = create_sequences_many(train_data_scaled, sequence_length_input, sequence_length_out, flatten = flatten,
                                                dtype = dtype) # LSTM should be flatten = False
        X_test, y_test = None, None
    return X_train, y_train, X_test, y_test, scaler
