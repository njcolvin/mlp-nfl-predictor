import numpy as np

def standardize(X):
    """
    Args:
        'X': numpy ndarray 
    Returns:
        'X_norm': normalized X also in numpy ndarray format
    """
    X_norm=(X-np.mean(X,axis=0))/np.std(X,axis=0)
    return X_norm