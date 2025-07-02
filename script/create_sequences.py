import numpy as np
def create_sequences(data: np.array, seq_length: int, flatten = False):
    """create_sequences of temporal data

    Args:
        data (np.array): original sequence data (date ascending order)
        seq_length (int): length of sequence to make

    Returns:
        tuple: X and y of seuqntial data
    """
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  
        y.append(data[i + seq_length])  
    if flatten:
        X = np.array(X)
        y = np.array(y)
        return X.reshape(X.shape[0],-1).astype('float16'), y.reshape(y.shape[0],-1).astype('float16')
    else:
        return np.array(X).astype('float16'), np.array(y).astype('float16')

def create_sequences_many(data: np.array, 
                           seq_length_input: int=3, 
                           seq_length_out: int=3, 
                           flatten: bool = False,
                           dtype  = 'float16'):
    
    """
    시계열 데이터를 입력(X)과 출력(y)으로 나누는 함수.

    Args:
        data (np.array): 정렬된 시계열 데이터 (날짜 순서 오름차순)
        seq_length (int): 입력 시퀀스 길이 (예: 30일)
        output_seq_length (int): 출력 시퀀스 길이 (예: 60일)
        flatten (bool): True일 경우 2D 배열로 평탄화하여 반환

    Returns:
        tuple: 입력(X)과 출력(y) 시퀀스 배열
    """
    X, y = [], []
    for i in range(len(data)- (seq_length_input + seq_length_out) + 1):
        feature = data[i : i+seq_length_input]
        target = data[i+seq_length_input : i+seq_length_input+seq_length_out]
        X.append(feature)
        y.append(target)
    if flatten:
        X = np.array(X)
        y = np.array(y)
        return X.reshape(X.shape[0],-1).astype(dtype), y.reshape(y.shape[0],-1).astype(dtype)
    else:
        return np.array(X).astype(dtype), np.array(y).astype(dtype)