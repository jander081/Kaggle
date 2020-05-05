# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.feature_selection import SelectFromModel,RFE, chi2

from sklearn.metrics import (mean_squared_error, r2_score, recall_score, confusion_matrix, classification_report, log_loss, precision_score, accuracy_score,f1_score,roc_auc_score)

from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


#------------------------------------------------------------
svr_params = {              
#               'gamma': [1e-5, 1e-1], 
                'C': [0.1, 1.0],
              'epsilon': [1e-5, 0.1]
}

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from skopt import BayesSearchCV
from sklearn.svm import SVR

class BayesSVR(SVR):
    
    def __init__(self, kernel='rbf', gamma='auto_deprecated',
                 coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, 
                 shrinking=True, cache_size=200, verbose=False, 
                 intervals=None):
        
        self.intervals=intervals
        self.model=SVR()
        
        super(BayesSVR, self).__init__(kernel=kernel, 
                                       gamma=gamma,
                                       coef0=coef0, tol=tol, C=C, 
                                       epsilon=epsilon,
                                       shrinking=shrinking,
                                       cache_size=cache_size, 
                                       verbose=verbose )
       
    def fit(self, X, y):
                
#         X_ = X.reset_index(drop=True)
#         y_ = pd.DataFrame(y)

#         X_tune_merge = pd.concat([X_, y_], axis=1)   
#         X_tune = X_tune_merge.sample(n=100, random_state=181)
#         y_tune = X_tune.iloc[:, -1]
#         X_tune = X_tune.iloc[:, :-1]
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=81)
        bayes =  BayesSearchCV(self.model, self.intervals, n_iter=5, n_jobs=-1, 
                               cv = kfold, verbose=0, random_state=82)

#         bayes.fit(X_tune, y_tune)
        bayes.fit(X, y)
        # bayes.best_params_.update( {'random_state': 183} )
        parameters = bayes.best_params_
        
        super(BayesSVR, self).__init__(**parameters,  shrinking=False)
        
        # Return the Regressor
        super(BayesSVR, self).fit(X, y)
                                     
        return self
    
    def predict(self, X):
        
        y = super(BayesSVR, self).predict(X)
       
        return np.asarray(y, dtype=np.float64)
    
    
#------------------------------------------------------------


rf_params = {'n_estimators':[100, 400], 
             
#             'max_depth': [3], 
#              'min_samples_split': [2, 3], 
#              'min_samples_leaf': [1, 3], 
             'min_weight_fraction_leaf':[0.0, 1e-3], 
# #              max_features='auto', 
#              'max_leaf_nodes': [40, 80], 
#              'min_impurity_decrease':[0.0, 1e-15], 
#              'min_impurity_split':[1e-8, 1e-5]
}

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor


class BayesRandomForest(RandomForestRegressor):
    
    def __init__(self, n_estimators=100, criterion='mse', 
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 min_weight_fraction_leaf=0.0, max_features='auto', 
                 max_leaf_nodes=None, min_impurity_decrease=0.0, 
                 min_impurity_split=None, 
                 n_jobs=-1, random_state=81, verbose=0,
                 intervals=None):
        
        self.intervals=intervals
        self.model=RandomForestRegressor()
        
        super(BayesRandomForest, self).__init__(n_estimators=n_estimators, 
                                               max_depth=max_depth, min_samples_split=min_samples_split, 
                                               min_samples_leaf=min_samples_leaf,
                                               min_weight_fraction_leaf=min_weight_fraction_leaf, 
                                               max_features=max_features,
                                               max_leaf_nodes=max_leaf_nodes,
                                               min_impurity_decrease=min_impurity_decrease,
                                               min_impurity_split=min_impurity_split,
                                               n_jobs=n_jobs, random_state=random_state, 
                                               verbose=verbose )
       
    def fit(self, X, y):
        
#         X_ = X.reset_index(drop=True)
#         y_ = pd.DataFrame(y)

#         X_tune_merge = pd.concat([X_, y_], axis=1)   
#         X_tune = X_tune_merge.sample(n=200, random_state=181)
#         y_tune = X_tune.iloc[:, -1]
#         X_tune = X_tune.iloc[:, :-1]
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=81)
        bayes =  BayesSearchCV(self.model, self.intervals, n_iter=5, n_jobs=-1, 
                               cv = kfold, verbose=0, random_state=82)

#         bayes.fit(X_tune, y_tune)
        bayes.fit(X, y)
        # bayes.best_params_.update( {'random_state': 183} )
        parameters = bayes.best_params_
        
        super(BayesRandomForest, self).__init__(**parameters)
        
        # Return the Regressor
        super(BayesRandomForest, self).fit(X, y)
                                     
        return self
    
    def predict(self, X):
        
        y = super(BayesRandomForest, self).predict(X)
       
        return np.asarray(y, dtype=np.float64)
    
#------------------------------------------------------------

lgb_params = {
              'num_leaves':[20, 50], 
              'max_depth': [3, 5], 
#               'learning_rate':[.0999, 0.101], 
              'n_estimators': [50, 150], 
              'subsample_for_bin':[175000, 225000],             
              'min_split_gain': [0.0, 1e-4], 
#               'min_child_weight': [1e-5, 1e-3], 
#               'min_child_samples': [18, 22], 
              'subsample':[0.8, 1.0], 
#               'subsample_freq': [0, 1], 
#               'colsample_bytree': [0.98, 1.0], 
              'reg_alpha': [0.0, 1e-7],
              'reg_lambda': [1e-3, 0.1]
}

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from skopt import BayesSearchCV
from lightgbm import LGBMRegressor

class BayesLGBMRegressor(LGBMRegressor):
    
    '''Unable to accept a dict as an input for some reason'''
    
    def __init__(self, num_leaves=31, max_depth=-1, learning_rate=0.1, 
                 n_estimators=100, subsample_for_bin=200000, 
                 objective=None, class_weight=None, 
                 min_split_gain=0.0, min_child_weight=0.001, 
                 min_child_samples=20, subsample=1.0, subsample_freq=0, 
                 colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, 
                 random_state=81, n_jobs=-1, intervals=None):
        
        self.intervals=intervals
        self.model=LGBMRegressor()
        
        super(BayesLGBMRegressor, self).__init__(num_leaves=num_leaves, 
                                                max_depth=max_depth, learning_rate=learning_rate, 
                                                n_estimators=n_estimators, 
                                                subsample_for_bin=subsample_for_bin, 
                                                objective=objective, 
                                                class_weight=class_weight, 
                                                min_split_gain=min_split_gain, 
                                                min_child_weight=min_child_weight, 
                                                min_child_samples=min_child_samples, 
                                                subsample=subsample, subsample_freq=subsample_freq, 
                                                colsample_bytree=colsample_bytree, 
                                                reg_alpha=reg_alpha, reg_lambda=reg_lambda, 
                                                random_state=random_state, n_jobs=n_jobs )

    def fit(self, X, y):
        
#         X_ = X.reset_index(drop=True)
#         y_ = pd.DataFrame(y)

#         X_tune_merge = pd.concat([X_, y_], axis=1)   
#         X_tune = X_tune_merge.sample(n=200, random_state=181)
#         y_tune = X_tune.iloc[:, -1]
#         X_tune = X_tune.iloc[:, :-1]
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=81)
        bayes =  BayesSearchCV(self.model, lgb_params, n_iter=5, n_jobs=-1, 
                               cv = kfold, verbose=0, random_state=82)

#         bayes.fit(X_tune, y_tune)
        bayes.fit(X, y)
        bayes.best_params_.update( {'random_state': 183} )
        parameters = bayes.best_params_
        
        super(BayesLGBMRegressor, self).__init__(**parameters)
        
        # Return the Regressor
        super(BayesLGBMRegressor, self).fit(X, y)
                                     
        return self
    
    def predict(self, X):
        
        y = super(BayesLGBMRegressor, self).predict(X)
       
        return np.asarray(y, dtype=np.float64)
    
#------------------------------------------------------------

xgb_params = {
        'learning_rate': (0.03, 0.1), 
        'min_child_weight': (1, 4),
        'max_depth': (3, 4),
#         'max_delta_step': (0, 2),
        'subsample': (0.8, 1.0),
        'colsample_bytree': (0.8, 1.0),
#         'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1.5),
        'reg_alpha': (1e-9, 1.0),
        'gamma': (1e-9, 0.1),
        'n_estimators': (150, 500)}

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from skopt import BayesSearchCV
from xgboost import XGBRegressor

class BayesXGBRegressor(XGBRegressor):
    
    def __init__(self, max_depth=3, learning_rate=0.1, 
                 n_estimators=100, 
                 n_jobs=-1,gamma=0, 
                 min_child_weight=1, max_delta_step=0, subsample=1, 
                 colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, 
                 reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
                 random_state=81, missing=None,
                 intervals=None):
        
        self.intervals=intervals
        self.model=XGBRegressor()
        
        super(BayesXGBRegressor, self).__init__(max_depth=max_depth, 
                                                learning_rate=learning_rate, 
                                                n_estimators=n_estimators, 
                                                n_jobs=n_jobs,gamma=gamma, 
                                                min_child_weight=min_child_weight, 
                                                max_delta_step=max_delta_step, 
                                                subsample=subsample, 
                                                colsample_bytree=colsample_bytree, 
                                                colsample_bylevel=colsample_bylevel, 
                                                reg_alpha=reg_alpha,reg_lambda=reg_lambda,
                                                scale_pos_weight=scale_pos_weight,
                                                base_score=base_score,
                                                random_state=random_state,
                                                missing=missing)

    def fit(self, X, y):
        
#         X_ = X.reset_index(drop=True)
#         y_ = pd.DataFrame(y)

#         X_tune_merge = pd.concat([X_, y_], axis=1)   
#         X_tune = X_tune_merge.sample(n=200, random_state=181)
#         y_tune = X_tune.iloc[:, -1]
#         X_tune = X_tune.iloc[:, :-1]
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=81)
        bayes =  BayesSearchCV(self.model, self.intervals, n_iter=5, n_jobs=-1, 
                               cv = kfold, verbose=0, random_state=82)

#         bayes.fit(X_tune, y_tune)
        bayes.fit(X, y)
        bayes.best_params_.update( {'random_state': 183} )
        parameters = bayes.best_params_
        
        super(BayesXGBRegressor, self).__init__(**parameters)
        
        # Return the Regressor
        super(BayesXGBRegressor, self).fit(X, y)
                                     
        return self
    
    def predict(self, X):
        
        y = super(BayesXGBRegressor, self).predict(X)
       
        return np.asarray(y, dtype=np.float64)
    
    

#------------------------------------------------------------
#------------------------------------------------------------


from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone


class StackingCVRegressorAveraged(BaseEstimator, RegressorMixin, TransformerMixin):
    
    def __init__(self, regressors, meta_regressor, n_folds=5):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds

    def fit(self, X, y):
        self.regr_ = [list() for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        for i, clf in enumerate(self.regressors):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(clf)
                self.regr_[i].append(instance)

                instance.fit(X.iloc[train_idx], y[train_idx])
                y_pred = instance.predict(X.iloc[holdout_idx])
                out_of_fold_predictions[holdout_idx, i] = y_pred

        self.meta_regr_.fit(out_of_fold_predictions, y)

        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([r.predict(X) for r in regrs]).mean(axis=1)
            for regrs in self.regr_
        ])
        return self.meta_regr_.predict(meta_features)
        

#------------------------------------------------------------
class AveragingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors):
        self.regressors = regressors

    def fit(self, X, y):
        self.regr_ = [clone(x) for x in self.regressors]
        
        # Train base models
        for regr in self.regr_:
            regr.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])
        return np.mean(predictions, axis=1)
    
#------------------------------------------------------------
class StackingCVRegressorRetrained(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors, meta_regressor, n_folds=5, use_features_in_secondary=False):
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        self.n_folds = n_folds
        self.use_features_in_secondary = use_features_in_secondary
        
    def fit(self, X, y):
        self.regr_ = [clone(x) for x in self.regressors]
        self.meta_regr_ = clone(self.meta_regressor)

        kfold = KFold(n_splits=self.n_folds, shuffle=True)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.regressors)))

        # Create out-of-fold predictions for training meta-model
        for i, regr in enumerate(self.regr_):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(regr)
                instance.fit(X.iloc[train_idx], y[train_idx])
                out_of_fold_predictions[holdout_idx, i] = instance.predict(X[holdout_idx])

            # Train meta-model
            if self.use_features_in_secondary:
                self.meta_regr_.fit(np.hstack((X, out_of_fold_predictions)), y)
            else:
                self.meta_regr_.fit(out_of_fold_predictions, y)

            # Retrain base models on all data
            for regr in self.regr_:
                regr.fit(X, y)

            return self

    def predict(self, X):
        meta_features = np.column_stack([
            regr.predict(X) for regr in self.regr_
        ])

        if self.use_features_in_secondary:
            return self.meta_regr_.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_regr_.predict(meta_features)
#------------------------------------------------------------



#------------------------------------------------------------

'''Notes: built-in super() function, which is a function for delegating method calls to some class in the instance’s ancestor tree. For our purposes, think of super() as a generic instance of our parent class.

http://flennerhag.com/2017-01-08-Recursive-Override/



'''

#------------------------------------------------------------
#------------------------------------------------------------


#------------------------------------------------------------


#------------------------------------------------------------
