# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin


#------------------------------------------------------------


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
    
    """
    DataFrame Wrapper around StandardScaler; Recursive override
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        super(StandardScalerDf, self).__init__(copy=copy,
                                               with_mean=with_mean,
                                               with_std=with_std)

    def transform(self, X, y=None):
        z = super(StandardScalerDf, self).transform(X.values)
        return pd.DataFrame(z, index=X.index, columns=X.columns)



#------------------------------------------------------------



class SelectFeatures(BaseEstimator, TransformerMixin):
    
    """
    Used with Kbins to select features with sufficient cardinality. Could
    probably just join this with kbins
    
    parameters
    ----------
    feat_list: list, default=None
        if you want a particular subset. This works as a feeder into the KBins 
        transformer
    
    
    """
    
    def __init__(self, val_count=50, categorical=False, feat_list=None):
        self.val_count = val_count
        self.categorical = categorical
        self.feat = pd.DataFrame()
        self.feat_list = feat_list
        
    
    def numeRical(self, X):
        # if there are no categoricals
        for col in X.columns:
                # if val_count threshold is met
                if len(X[col].value_counts()) > self.val_count:
                    # relabel
                    X[col + '_bin'] = X[col]
                    # add it to the dataframe
                    self.feat = pd.concat([self.feat, X[col + '_bin']], axis=1)
        return self.feat
    
    
    def cateGorical(self, X):
        # if there are  categoricals
        for col in X.columns:
                # Do not need to relabel the objs since they will be converted to freqs 
                # prior to binning
            if len(X[col].value_counts()) > self.val_count: 
                self.feat = pd.concat([self.feat, X[col]], axis=1)   
        return self.feat
    
    
    def fit(self, X, y=None):
        return self


    def transform(self, X):
        assert isinstance(X, pd.DataFrame), "must be pandas"
        
         
        if self.feat_list:
            assert isinstance(self.feat_list, list), "must be a list"
            for col in self.feat_list:
                X[col + '_bin'] = X[col]
                # watch the double brackets or you'll lose the colname
                self.feat = pd.concat( [self.feat, X[[col + '_bin']]] )
                            
        elif self.categorical:
            self.feat = self.cateGorical(X)             
        else:
            self.feat = self.numeRical(X)
            
        return self.feat

    
#------------------------------------------------------------
from sklearn.preprocessing import KBinsDiscretizer

class KBins(KBinsDiscretizer):
    
    """DataFrame Wrapper around KBinsDiscretizer. Sometimes this will throw 
    the monotonically increase/decrease error. You can either reduce bins 
    or modify the selected features by value counts (increase)
    
    It comes out of the parent function as ordinal. The wrapper converts it to 
    categorical.
    
    Note: these wrappers are hard to tailor beyond the basic. It may be better
    to tailor the prior transformer 
    """

    def __init__(self, n_bins=5, encode='ordinal', strategy='quantile'):
        super(KBins, self).__init__(n_bins=n_bins,
                                    encode=encode,
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
 

class FreqFeatures(BaseEstimator, TransformerMixin):
    
    """
    returns a dict for freqs. This can then be mapped to 
    any col to create freq feature. Must be run prior to freq_group
    """
       
    def __init__(self, val_count=50):
        self.val_count = val_count
        self.drops = []
        
    def make_dict(self, col):
        
        df = pd.DataFrame(self.data[col].value_counts())
        df.reset_index(level=0, inplace=True)
        df.rename(columns={'index': 'key', col: 'value'}, inplace=True)
        df_dict = defaultdict(list)
        for k, v in zip(df.key, df.value):
            df_dict[k] = (int(v))
        return df_dict
    
    @staticmethod
    def reduce(x,y):
        if x <= 10:
            return 'rare'
        elif x <= 50:
            return 'infrequent'
        elif x <= 100:
            return 'less common'
        else:
            return y

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.data = X
        
        assert isinstance(self.data, pd.DataFrame), 'pls enter dframe'
        
        for col in self.data.columns:
            dict_ = self.make_dict(col)        
            freq_vec = self.data[col].map(dict_)
            
            if len(set(self.data[col])) > 2000:
                self.drops.append(col)
                self.data.drop([col], axis=1, inplace=True)
                self.data[col + '_freq'] = freq_vec
                
            elif len(set(self.data[col])) > 100:
                y = self.data[col]
                vectfunc = np.vectorize(self.reduce,cache=False)
                vec = np.array(vectfunc(freq_vec,y))
                
                self.data[col + '_freq'] = freq_vec
                self.data[col + '_reduce'] = vec
                self.data.drop([col], axis=1, inplace=True)
                
        return self.data
                


'''Notes: built-in super() function, which is a function for delegating method calls to some class in the instance’s ancestor tree. For our purposes, think of super() as a generic instance of our parent class.
http://flennerhag.com/2017-01-08-Recursive-Override/
'''




        
        
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------




