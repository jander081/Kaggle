import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.feature_selection import chi2, SelectKBest

import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

from numpy.random import seed
seed(123)
from tensorflow import set_random_seed
set_random_seed(234)
train = pd.read_csv("train (1).csv")
test = pd.read_csv("test (1).csv")
full = train.append(test, ignore_index = True )

titanic = full[:891]
del train , test
titanic.columns

def findnan(df, n):    
    '''n = "column"'''
    count = []
    for i in df[n]:
        if pd.isnull(i) == True:
            count.append(i)
    print('{0} = {1}'.format(n, len(count)))
    
def allnans(df):
    for i in df.columns:
        findnan(df, i)
        
full.Embarked = full.Embarked.fillna('S')
embarked = pd.get_dummies(full.Embarked, prefix='embarked', 
                          drop_first=True)

sex = pd.get_dummies(full.Sex,drop_first=True)

pclass = pd.get_dummies(full.Pclass, prefix='pclass',
                       drop_first=True)

full.Fare = full.Fare.fillna(full.Fare.mean())
