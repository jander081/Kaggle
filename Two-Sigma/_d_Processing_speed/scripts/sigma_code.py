'''
----------------------------------------------------------
PROCESSING FUNCS
----------------------------------------------------------
'''

import pandas as pd
import numpy as np


def mem_usage(pandas_obj):
    
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

def d_type_usage(dframe):
    
    for dtype in ['float','int','object']:
        selected_dtype = dframe.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,
              mean_usage_mb))



'''
----------------------------------------------------------
ENGINEERING FUNCS
----------------------------------------------------------
'''
import datetime as dt



def ticker_ex():
    """
    quick way to separate the extensions - not ready.. paste to use
    """
    ext_vec = []
    asst_vec = []
    for i in df.assetCode:
        try:
            asst_vec.append(i.split('.')[0])
            # can do list comprehension for the first term
            ext_vec.append(i.split('.')[1])
        except IndexError:
            # some extension don't exist
            ext_vec.append(np.nan)
    df['ext'] = ext_vec
    df['assetTicker'] = asst_vec




def timestamp_conv(x):
    
    term = x.split()[0]
    return dt.datetime.strptime(term, '%Y-%m-%d')
  
    
    df_n['Date'] = df_n.time.apply(timestamp_conv)
    df_n.drop(['time'], axis=1, inplace=True)
    
    

import random

def universe_feat(X, n=8):
    '''creates universe feat for data testing'''
    v = []
    for i in range(0, X.shape[0]):
        lst = np.ones(n).tolist() + [0]
        u = random.choice(lst)
        v.append(u)
    return np.array(v)

'''
----------------------------------------------------------
OTHER
----------------------------------------------------------
'''

def custom_metric(date, pred_proba, num_target, universe):
    y = pred_proba*2 - 1
    r = num_target.clip(-1,1) # get rid of outliers
    x = y * r * universe
    result = pd.DataFrame({'day' : date, 'x' : x})
    x_t = result.groupby('day').sum().values
    return np.mean(x_t) / np.std(x_t)

