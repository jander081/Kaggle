# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime


#------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin

class TypeSelector(BaseEstimator, TransformerMixin):
    
    '''np.object, np.number, np.bool_'''
    
    def __init__(self, dtype1, dtype2=None, dtype3=None):
        self.dtype1 = dtype1
        self.dtype2 = dtype2
        self.dtype3 = dtype3

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        assert isinstance(X, pd.DataFrame), "Gotta be Pandas"
        
        if self.dtype3 != None:
            
            output = (X.select_dtypes(include=[self.dtype1]),
                   X.select_dtypes(include=[self.dtype2]),
                   X.select_dtypes(include=[self.dtype3]))
            
        elif self.dtype2 != None:
            output = (X.select_dtypes(include=[self.dtype1]),
                   X.select_dtypes(include=[self.dtype2]))
            
        else:
            
            output = (X.select_dtypes(include=[self.dtype1]))
            
        return output
        

#------------------------------------------------------------

from sklearn.preprocessing import StandardScaler 

class StandardScalerDf(StandardScaler):
    """DataFrame Wrapper around StandardScaler; Recursive override"""

    def __init__(self, copy=True, with_mean=True, with_std=True):
        super(StandardScalerDf, self).__init__(copy=copy,
                                               with_mean=with_mean,
                                               with_std=with_std)

    def transform(self, X, y=None):
        z = super(StandardScalerDf, self).transform(X.values)
        return pd.DataFrame(z, index=X.index, columns=X.columns)


#------------------------------------------------------------

from fancyimpute import SoftImpute

class SoftImputeDf(SoftImpute):
    """DataFrame Wrapper around SoftImpute"""

    def __init__(self, shrinkage_value=None, convergence_threshold=0.001,
                 max_iters=100,max_rank=None,n_power_iterations=1,init_fill_method="zero",
                 min_value=None,max_value=None,normalizer=None,verbose=True):
        
        super(SoftImputeDf, self).__init__(shrinkage_value=shrinkage_value, 
                                           convergence_threshold=convergence_threshold,
                                           max_iters=max_iters,max_rank=max_rank,
                                           n_power_iterations=n_power_iterations,
                                           init_fill_method=init_fill_method,
                                           min_value=min_value,max_value=max_value,
                                           normalizer=normalizer,verbose=False)

    

    def fit_transform(self, X, y=None):
        
        assert isinstance(X, pd.DataFrame), "Must be pandas dframe"
        
        for col in X.columns:
            if X[col].isnull().sum() < 10:
                X[col].fillna(0.0, inplace=True)
        
        z = super(SoftImputeDf, self).fit_transform(X.values)
        return pd.DataFrame(z, index=X.index, columns=X.columns)


#------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin

class SelectFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, val_count=50, categorical=False):
        self.val_count = val_count
        self.categorical = categorical
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        feat = pd.DataFrame()
        
        if self.categorical==False:           
            for col in X.columns:
                if len(X[col].value_counts()) > self.val_count:              
                    X[col + '_bin'] = X[col]
                    feat = pd.concat([feat, X[col + '_bin']], axis=1)
        else:
            for col in X.columns:
                if len(X[col].value_counts()) > self.val_count: 
                    X[col + '_freq'] = X[col]
                    feat = pd.concat([feat, X[col + '_freq']], axis=1)                    
        return feat
    
#------------------------------------------------------------
from sklearn.preprocessing import KBinsDiscretizer

class KBins(KBinsDiscretizer):
    
    """DataFrame Wrapper around KBinsDiscretizer"""

    def __init__(self, n_bins=5, encode='onehot', strategy='quantile'):
        super(KBins, self).__init__(n_bins=n_bins,
                                    encode='ordinal',
                                    strategy=strategy)                               
        
       
    def transform(self, X, y=None):
        
        assert isinstance(X, pd.DataFrame), "Must be pandas dframe"
        
        
        z = super(KBins, self).transform(X)
        binned = pd.DataFrame(z, index=X.index, columns=X.columns)
        binned = binned.applymap(lambda x: 'category_' + str(x))
#         final = pd.concat([X, binned], axis=1)        
        return binned


    
#------------------------------------------------------------

import re

class RegImpute(BaseEstimator, TransformerMixin):
    
    '''consider adding methods to check for special characters or return
    indices for nans, since nans can be different types. If bool, shut off
    regex'''
    
    def __init__(self, regex=True):
        self.regex = regex
        
    def find_nulls(self, X, y=None):
        '''this returns all dframe indices with nans. Useful to determine
        type of null'''
        return pd.isnull(X).any(1).nonzero()[0]
    
    def null_cols(self, X, y=None):
        '''Prints list of null cols with number of nulls'''
        null_columns=X.columns[X.isnull().any()]
        print(X[null_columns].isnull().sum())
              
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        
        for col in X.columns:
            X[col].fillna(X[col].mode().iloc[0], inplace=True)
            
        if self.regex == True:
            X = X.applymap(lambda x: re.sub(r'\W+', '', x)) 
            
        return X

#------------------------------------------------------------
 
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin

class FreqFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, val_count=50):
        self.val_count = val_count
        
    def make_dict(self, col, X):
        
        df = pd.DataFrame(X[col].value_counts())
        df.reset_index(level=0, inplace=True)
        df.rename(columns={'index': 'key', col: 'value'}, inplace=True)
        df_dict = defaultdict(list)
        for k, v in zip(df.key, df.value):
            df_dict[k] = (int(v))
        return df_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        
        for col in X.columns:
            X[col] = X[col].map(self.make_dict(col, X))
                
        return X
    
    
#------------------------------------------------------------

from collections import defaultdict

def make_dict(col, dframe):
    
    '''returns a dict for freqs. This can then be mapped to 
    any col to create freq feature. Must be run prior to freq_group'''
    
    
    df = pd.DataFrame(dframe[col].value_counts())
    df.reset_index(level=0, inplace=True)
    df.rename(columns={'index': 'key', col: 'value'}, inplace=True)
    df_dict = defaultdict(list)
    for k, v in zip(df.key, df.value):
        df_dict[k] = (int(v))
    return df_dict


#------------------------------------------------------------

def freq_group(freq, _dict, rare, infrequent, less_common):
    
    '''run as lambda on col; feature value aggregator'''
    
    rev_dict = {v:k for k, v in _dict.items()}
    
    if freq <= rare:
        string = 'rare'
    elif freq > rare and freq <= infrequent:
        string = 'infrequent'
    elif freq > infrequent and freq <= less_common:
        string = 'less common'
    else:
        string = rev_dict[freq]
    return(string)

#------------------------------------------------------------


'''Notes: built-in super() function, which is a function for delegating method calls to some class in the instance’s ancestor tree. For our purposes, think of super() as a generic instance of our parent class.

http://flennerhag.com/2017-01-08-Recursive-Override/



'''

#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------


#------------------------------------------------------------
