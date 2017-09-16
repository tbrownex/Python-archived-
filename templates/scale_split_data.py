''' Assumptions are:
     1) incoming data is in a pandas dataframe, not numpy array
     2) The last column holds the Target
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import MinMaxScaler
from sklearn.preprocessing   import StandardScaler
import pandas as pd
import numpy  as np

def split(data, val_pct, test_pct):
    assert test_pct > 0 or val_pct > 0, "must specify either a test pct or validation pct"
    sv_cols = data.columns
    
    train, test = train_test_split(data, test_size=test_pct)
    test.columns = sv_cols
    if val_pct > 0:
        train, val = train_test_split(train, test_size=val_pct)
        train.columns = sv_cols
        val.columns   = sv_cols
        return train, val, test
    else:
        train.columns = sv_cols
        return train, None, test
    
    '''
    MinMaxA: Linear scaling from 0, +1
    MinMaxB: Linear scaling from -1, +1
    Std:     Gaussian distribution with mean = 0 and StdDev = 1
    tanh:    sigmoid-looking scaling from -1, +1; see here for reference:
http://stats.stackexchange.com/questions/7757/data-normalization-and-standardization-in-neural-networks
    '''
def normalize(data, method):
    assert isinstance(data, pd.DataFrame),   "data must be pandas DataFrame"
    assert method in ['MinMaxA','MinMaxB', 'Std', 'tanh'],  "Invalid scaling method"
    sv_cols = data.columns
    if method =='MinMaxA':
        min_max_scaler = MinMaxScaler()
        data           = pd.DataFrame(min_max_scaler.fit_transform(data))
        data.columns   = sv_cols
        scale_range    = min_max_scaler.data_range_[-1]
        scale_min      = min_max_scaler.data_min_[-1]
        return data, scale_range, scale_min
    elif method =='MinMaxB':
        arr        = data.values
        width      = np.ptp(arr, axis=0)
        scale_min  = np.min(arr, axis=0)
        norm       = (2 * (arr - scale_min) / width) -1
        data       = pd.DataFrame(norm, columns=sv_cols)
        return data, width[-1], scale_min[-1]
    elif method == 'Std':
        std_scaler     = StandardScaler()
        data           = pd.DataFrame(std_scaler.fit_transform(data))
        data.columns   = sv_cols
        avg            = std_scaler.mean_[-1]
        std            = np.sqrt(std_scaler.var_[-1])
        return data, std, avg
    else:
        arr = data.values
        std = np.std(arr,axis=0)
        avg = np.mean(arr,axis=0)
        Z = (arr - avg) / std
        norm = 0.5 * (np.tanh( Z * .01 ) + 1)
        data = pd.DataFrame(norm, columns=sv_cols)
        return data, std[-1], avg[-1]

# "a" and "b" have different meanings depending on "method":
# MinMaxA and B:   a is the range; b is the minimum
#    Std:   a is the StdDev; b is the mean
#   tanh:   same as Std
def denormalize(data, method, a, b):
    if method == 'MinMaxA':
        return data * a + b
    elif method == 'MinMaxB':
        return (data + 1) * a / 2 + b
    elif method == 'Std':
        return data * a + b
    else:
        Z = np.arctanh(data * 2 - 1) * 100
        data = Z * a + b
    return data