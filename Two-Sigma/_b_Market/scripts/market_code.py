
import datetime as dt
import pandas as pd
import numpy as np
import pandas_datareader as web
import gc



####################################################################    
#                   ENGINEERING FUNCTIONS                             #
####################################################################


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




def trainFeats(X, drops=list):
        
        feats = []
        for col in X.columns:
            if col not in drops:
                feats.append(col)
                
        return feats


####################################################################    
#                   DATA GENERATOR                                 #
####################################################################

from pandas_datareader._utils import RemoteDataError


def marketDataGen(asset_list, start, end=None):
    
    """may want to use a subset of assets. Make sure assets are cleaned.
    $ sub = [i.split('.')[0] for i in sub_asst]. Asset lists are kept in 
    the output folder.
    """
    assert isinstance(asset_list, list), 'must input a list of assets'
    market = pd.DataFrame()
    for i in asset_list:
    
        try:
            vec = web.DataReader(i, 'yahoo', start=start, end=end)
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
    
    # convert timestamp to int for dataframe splits - leave out for now
    
    
    # make it pretty
    cols = ['asset',u'Open',u'Close',u'Volume',
 'returns_close_raw','returns_open_raw','returns_close_raw10',     
'returns_open_raw10','returns_open_raw10_next']
    
    return market[cols]



####################################################################    
#                  TECHNICAL INDICATOR FEATURES                    #
####################################################################


# RSI


'''
----------------------------------------------------------
OTHER
----------------------------------------------------------
'''

def custom_metric(date, pred_proba, num_target, universe):
    y = pred_proba*2 - 1
    r = num_target#.clip(-1,1) # get rid of outliers
    x = y * r * universe
    result = pd.DataFrame({'day' : date, 'x' : x})
    x_t = result.groupby('day').sum().values
    return np.mean(x_t) / np.std(x_t)


####################################################################    
#                  PIPELINE FEAUTURES                   #
####################################################################


def pipe(original):
    """
    Need to break this down 
    
    """
    class PipeInto(object):
        data = {'function': original}

        def __init__(self, *args, **kwargs):
            self.data['args'] = args
            self.data['kwargs'] = kwargs

        def __rrshift__(self, other):
            return self.data['function'](
                other, 
                *self.data['args'], 
                **self.data['kwargs']
            )

    return PipeInto



import datetime as dt

@pipe
def exTractTime(X, col, atts=list, mthds=None):

    mod = getattr(X, col)

    for i in range(len(atts)):
    
        X[atts[i]] = getattr(mod.dt, atts[i])
    
    while mthds:
        for i in range(len(mthds)):
    
            f = getattr(mod.dt, mthds[i])
            X[mthds[i]] = f()
        break
        
    return X


@pipe
def toCatFeat(X, feats=list):
    for i in feats:
        colName = i + '_obj'
        X[colName] = X[i].astype(str) + '_'
    return X



@pipe
def macdFeats(X):
    
    """
    If macd is negative/positive, the shorter term mv avg is below/above the longer 
    term and momentum is downward/upward - ish.
    
    If the signal is simply the ema of macd. If macd is less/greater than the signal,
    whether pos or neg, the momentum is bear/bullish
    
    Ex
    -----
    both signal and macd are negative. However, macd is noticably less
    negative than the signal. Interpretation: the asset has been in a short term down
    trend (short term down trend could be part of a long term down/up trend). However, 
    the momentum is bullish
    
    Metrics for indicators
    ------
    Crossover: macd is below/above signal
    
    Divergence: the price diverges by a certain threshold from macd. End of 
    current trend
    
    Dramatic Rise: macd is significantly above signal - overbought
       
    """
    
    # compute macd
    asset_close = X.groupby('assetCode')['close']
    ema_26 = asset_close.transform(lambda x: x.ewm(span=26).mean()).values
    ema_12 = asset_close.transform(lambda x: x.ewm(span=12).mean()).values
    macd = ema_12 - ema_26
    
    del asset_close, ema_12, ema_26
    
    # Need to define this feature for the groupby
    X['macd'] = macd
    sig =(
        X.groupby('assetCode')['macd'].
        transform(lambda x: x.ewm(span=9).
        mean()).
        values
        )
    X['sig'] = sig
    
    diff = macd - sig
    close = X['close'].values
    div = abs(close - macd)
    
    # Define the bools - explain thresholds
    X['macd_cross'] = np.where(macd > sig, 1, 0).astype('bool')
    X['macd_sharp_rise'] = np.where(diff > 0.09, 1, 0).astype('bool')
    X['macd_div'] = np.where(div > 75, 1, 0).astype('bool')
    
    del macd, sig, diff, close, div
    gc.collect()
    
    return X


    
@pipe
def bbSqueeze(X):
    
    '''price moves toward upper indicate bullish. This
    returns the distance from stock price to upper or lower 
    BB. If the distance turns negative, the price is above/below. 
    See indicator lambdas.
    
    Squeeze is calculate per asset. However, comparing it to the 6 month min is a bit
    more involved. For now, leave it external. It can be integrated into the func/class
    later if it is useful. Notice the shift on the rolling min(). The present squeeze value
    should not be included in the window we are comparing it with.
    '''
    
    asset_close = X.groupby('assetCode')['close']
    
    sd_20 = (
            asset_close.
            transform(lambda x: x.rolling(window=20).
            std()).
            values        
            )
    
    sma_20 = (
            asset_close.
            transform(lambda x: x.rolling(window=20).
            mean()).
            values
            )
    
    # convert everything to arrays   
    close = X['close'].values   
    U = sma_20 + (sd_20*2)    
    L = sma_20 - (sd_20*2) 
    squeeze = (U - L)/ sma_20
    
    del U, L, asset_close
        
    X['squeeze'] = squeeze
    # We want the nulls - hence min period
    sq_min = (
            X.groupby('assetCode')['squeeze'].
            transform(lambda x: x.rolling(window=126, min_periods=126).
            min().
            shift())
            )
    # Use to drop null, drop from training features
    X['sq_min'] = sq_min
    
    # Define bool
    X['low_vol'] = np.where(squeeze <= sq_min, 1, 0).astype('bool')

    del squeeze, close, sq_min
    gc.collect()
        
    return X

 
    
@pipe
def rsiFeats(X):    
    """
    May consider adding bools or binning
    """
    
    def rsi(x):    
        diff = x.diff()
        mask = diff < 0
        high = abs(diff.mask(mask)).fillna(0)
        low = abs(diff.mask(~mask)).fillna(0)
        pos_rsi = high.ewm(span=14).mean()
        neg_rsi = low.ewm(span=14).mean()
        
        del diff, mask, high, low
        
        return  pos_rsi / (pos_rsi + neg_rsi)
    
    X['rsi'] = (
                X.groupby('assetCode')['close'].
                transform(lambda x: rsi(x)).
                values
                )
    
    gc.collect()
    
    return X
   


####################################################################    
#                  TRANSFORMER                           #
####################################################################

from market_trans import *
from sklearn.pipeline import make_pipeline
import sys
paths = [
        '/Users/jacob/Desktop/docs/ML/_a_python/_1_code/notebooks/quickpipeline',
         '/Users/jacob/Desktop/docs/ML/_a_python/_1_code/notebooks/pd_feature_union' 
        ]

for path in paths:
    sys.path.append(path)
        
from quickpipe_mod import * 
from pandas_feature_union import *




def preprocPipe(X, binFeat):
    
    transformer_list = [ ('numeric', make_pipeline(TypeSelector(np.number),
                                QuickPipeline_mod()
                                )
                        ),('binned_features', make_pipeline(
                            TypeSelector(np.number),
                            SelectFeatures(feat_list=[binFeat]),
                            KBins(n_bins=5),
                            QuickPipeline_mod()
                             )
                        ), ('boolean_features', make_pipeline(
                                 TypeSelector(np.bool_),
                                QuickPipeline_mod(categorical_features=(TypeSelector(np.bool_).
                                                                        fit_transform(X).
                                                                        columns.
                                                                        tolist()
                                                                        ))
                                )
                       ),('categorical_features', make_pipeline(
                             TypeSelector(np.object),
                            QuickPipeline_mod() )
                            )                                           
                       ]
        
    pipe = make_pipeline(PandasFeatureUnion(transformer_list))
    print('pipeline created')
    return pipe
    





