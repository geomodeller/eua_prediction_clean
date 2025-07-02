import numpy as np
def generate_multivariate_samples(corr_matrix: np.array, n_samples:int = 1000):
    """generate_multivariate_samples

    Args:
        corr_matrix (np.array): correlation matrix among features
        n_samples (int, optional): number of samples. Defaults to 1000.

    Returns:
        np.array: correlated_samples
    """
    # Step 1: Cholesky decomposition of the correlation matrix
    L = np.linalg.cholesky(corr_matrix)
    
    # Step 2: Generate random samples from a standard normal distribution
    num_variables = corr_matrix.shape[0]
    standard_normal_samples = np.random.normal(size=(n_samples, num_variables))
    
    # Step 3: Apply the Cholesky factor to introduce correlations
    correlated_samples = standard_normal_samples @ L.T
    
    return correlated_samples