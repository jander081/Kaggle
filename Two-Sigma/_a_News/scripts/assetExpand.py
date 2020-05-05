import re
import pandas as pd
import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin


# ------------------------------------------------------------------

class AssetExpand(BaseEstimator, TransformerMixin):
    """
    Preprocesses the assetCodes column into a series of clean lists. 
    Fit_transform creates a tuple for each idx value 
    and its corresponding list (of assets). The len of the asset list is then
    used to generate a list of identical idx values of equal length. Additional 
    assets and there corresponding indices are then added in this manner and 
    eventually converted to cols in a dataframe.
    
    Example
    
    $ (0, 'AA', 'BB', 'CC')
    # returns
    $ [0, 0, 0] and ['AA', 'BB', 'CC']
    $ (1, 'DD', 'EE')
    # updates
    $ [0, 0, 0, 1, 1] and ['AA', 'BB', 'CC', 'DD', 'EE']
    
    Parameters
    -------------
    
    col: string, default='assetCodes'
        col identifier for expansion
        
    Attributes
    -------------
    These could be with the fit_transform method. These are the lists used
    as described in the example
    
    
    Notes
    -------------
    This is probably going to cause kernel issues. Continue to update with
    vector functions when possible. There should be a good example in the 
    thesis code
    
    """
     
        
    def __init__(self, col='assetCodes'):
        
        self.col = col
        self.asset_col = []
        self.idx_col = []


    @staticmethod
    def vec_func(x):
        return re.sub(r'[{}\']', '', x).replace(" ", "")
        
    def to_vec(self):

        # vector operations
        vectfunc = np.vectorize(self.vec_func,cache=False)
        vec = vectfunc(self.data[self.col])
                    
        return pd.Series([i.split(',') for i in vec.tolist()])
       
        

    def to_dframe(self):

        df_idx = pd.DataFrame({'index_col': self.idx_col, 
                                'assetCode': self.asset_col})
        # print(df_idx.head(2))
        # if you observe, you'll see repeating indices
        df = pd.concat([df_idx, self.data], axis=1, join='outer', 
                              join_axes=[df_idx.index])
        
        del df_idx, self.data, self.asset_col, self.idx_col
        
        df.drop([self.col, 'index_col'], axis=1, inplace=True)
        
        return  df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.data = X

        assert isinstance(self.data, pd.DataFrame), 'pls enter dframe'

        series = self.to_vec()

        for idx, asst in series.iteritems():

            self.asset_col.extend(asst) # tacking the assts on

            repeat_index = [int(idx)]*len(asst) # repeats it so many times

            self.idx_col.extend(repeat_index)
            
        del series

        updated_data = self.to_dframe()

        gc.collect()

        return updated_data
    
# ------------------------------------------------------------------
    
    
def preprocess2(dframe):
    
    '''Modification: this returns the news df for a merge. However, the merge
    should be on asset code, name, and date with null rows subsequently removed'''


    dframe['assetCodes'] = dframe['assetCodes'].apply(lambda x: re.sub(r'[{}\']', '', x).replace(" ", ""))
    # SOME WEIRD ERROR WHEN RUNNING A LAMBDA WITH X.SPLIT(','); WORKS FINE
    # ON MY LOCAL. THIS IS A WORKAROUND.
    feat = []
    for i in dframe['assetCodes']:
        new = i.split(',')
        feat.append(new)    
    dframe['assetCodes'] = feat
    del feat

    # EMBARRISING HOW MUCH FASTER THIS METHOD IS - LESSON LEARNED?
    asts = []
    idxs = []
    for idx, ast in dframe['assetCodes'].iteritems():
        asts.extend(ast)
        repeat_index = [int(idx)]*len(ast)
        idxs.extend(repeat_index)
    df_idx = pd.DataFrame({'index_col': idxs, 'assetCode': asts})
    del asts, idxs
    gc.collect()
    df_idx.head()

    df_idx.set_index('index_col', inplace=True)

    dframe_2 = pd.concat([df_idx, dframe], axis=1,
                     join='outer',join_axes=[df_idx.index]).reset_index(drop=True) 

    dframe_2.drop(['assetCodes'], axis=1, inplace=True)
    del df_idx, dframe
    gc.collect()

    dframe_2['date'] = pd.to_datetime(dframe_2.time).dt.date  # Add date column
    
    return(dframe_2)
    
    
