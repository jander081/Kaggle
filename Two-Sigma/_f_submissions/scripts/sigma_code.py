
import datetime as dt
import pandas as pd
import numpy as np
import pandas_datareader as web
'''
----------------------------------------------------------
ENGINEERING FUNCS
----------------------------------------------------------
'''


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


####################################################################    
#                   DATA GENERATOR                                 #
####################################################################

from pandas_datareader._utils import RemoteDataError


def marketDataGen(assets=list):
    
    """may want to use a subset of assets. Make sure assets are cleaned.
    $ sub = [i.split('.')[0] for i in sub_asst]. Asset lists are kept in 
    the output folder.
    """
    
    market = pd.DataFrame()
    for i in assets:
    
        try:
            vec = web.DataReader(i, 'yahoo', start='1/1/2013', end='1/1/2017')
            vec['asset'] = i
            vec['returns_close_raw'] = np.log(vec.Close/vec.Close.shift())
            vec['returns_open_raw'] = np.log(vec.Open/vec.Open.shift())
            vec['returns_open_raw10'] = np.log(vec.Open/vec.Open.shift(10))
            vec['returns_close_raw10'] = np.log(vec.Close/vec.Close.shift(10))
            vec['returns_open_raw10_next'] = np.log(vec.Open/vec.Open.shift(-10))
            market = pd.concat([market, vec])

        except RemoteDataError:
            print('remote error')


        except KeyError:
            print('key error')
            
    market.dropna(inplace=True) # there are a lot
    market.sort_index(inplace=True) # by trading days
    
    # make it pretty
    cols = ['asset',u'Open',u'Close',u'Volume',
 'returns_close_raw','returns_open_raw','returns_close_raw10',     
'returns_open_raw10','returns_open_raw10_next']
    
    return market[cols]















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

