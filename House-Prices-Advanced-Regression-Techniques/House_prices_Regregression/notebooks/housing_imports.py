import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt

import missingno as msno

#-----------sklearn----------------------------------------------------

from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.feature_selection import SelectFromModel,RFE, chi2

from sklearn.metrics import (mean_squared_error, r2_score, recall_score, confusion_matrix, classification_report, log_loss, precision_score, accuracy_score,f1_score,roc_auc_score)

from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

#---------------------------------------------------------------

from scipy.stats import chi2_contingency, boxcox
from skopt import BayesSearchCV
from fancyimpute import SoftImpute

from collections import defaultdict
from datetime import datetime
# from imblearn.over_sampling import SMOTE


#-----------MODELS----------------------------------------------------

from xgboost import XGBRegressor
from xgboost import plot_importance
import xgbfir

import datetime
import calendar 
import sys
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import re
import random
import pickle

import warnings
warnings.filterwarnings(action='ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


#----------------------------sys append-----------------------------------


path = '../code/'
sys.path.append(path + 'quickpipeline')
# sys.path.append(path + 'cleaners')
# sys.path.append(path + 'random_code')
sys.path.append(path + 'pd_feature_union')


from quickpipe_mod import *
# from quickpipeline import QuickPipeline
from pandas_feature_union import *

from housing_code import *
from housing_models import *

