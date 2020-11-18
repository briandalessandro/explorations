import pandas as pd
import numpy as np
import os
import ast
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import roc_auc_score, precision_score, zero_one_loss, recall_score

class ModelCVWrapper(object):
    
    def __init__(self, n_splits):
        self.cv_summary_df = None
        self.model_type = None
        self.n_splits = n_splits
        
        
    def _get_default_grid(self):
        
        if self.model_type == 'classification':
            rf_grid = {'n_estimators':[200,500], 'max_depth':[5,10], 'criterion':['entropy']}
            gbdt_grid = {'n_estimators':[100,200,500],'max_depth':[5,7], 
                    'learning_rate':[0.05,0.1]}
            grid_dict = {RandomForestClassifier():rf_grid, GradientBoostingClassifier(): gbdt_grid}            
        else:
            rf_grid = {'n_estimators':[200,500], 'max_depth':[5,10], 'criterion':['mse','mae']}
            gbdt_grid = {'n_estimators':[100,200,500],'max_depth':[5,7], 'loss':['ls'], 'alpha':[0.5],
                    'learning_rate':[0.05,0.1]}
            grid_dict = {RandomForestRegressor():rf_grid, GradientBoostingRegressor(): gbdt_grid}
            
        return grid_dict
        
    def get_model_cv(self, train_df, test_df, Y, feature_importance, thresholds,
                          grid_dict=None, print_status=True):
        '''
        Inputs:
        train_df is the training data, CV will be applied to it
        test_df is the out of sample 
        grid_dict has the format of : {sklearn.model_type: parameter_grid}
        feature_importance is a FeatureImportanceSummary object
        thresholds -> the feature importance thresholds you want to test. 

        Performs cross-validation for each model type, on different subsets of model features
        
        Automatically determines regression or classification based on target value
    
        Returns a dataframe of cross-validation results
        '''
        
        #Determine if Y is a continuous or discrete variable
        num_distinct_targets = len(train_df[Y].value_counts())
            
        if num_distinct_targets <= 10:
            self.model_type = 'classification'
        else:
            self.model_type = 'regression'
        
        if grid_dict is None:
            grid_dict = self._get_default_grid()


        kf = KFold(n_splits=self.n_splits, random_state=10, shuffle=False)

        #Get the index and create a list of features at a certain feat imp threshold
        ordered_x = feature_importance.feat_imp_df.x.values
        
        subsets = []
        for thresh in thresholds:
            subsets.append(ordered_x[0:feature_importance.feat_imp_index[thresh]])
                
    
        results = []
        for i, subset in enumerate(subsets):
            for model in grid_dict:
                
                if print_status:
                    print('Running Treshold {}'.format(thresholds[i]))
                
                
                #Run a grid search CV on the particular classifier
                mod = GridSearchCV(model, grid_dict[model], cv=kf, scoring='neg_mean_squared_error')
                mod.fit(train_df[subset], train_df[Y])
        
                #Get Mean-Squared Error on ttest set
                test_err = ((mod.best_estimator_.predict(test_df[subset]) - test_df[Y])**2).mean()
            
                #Append results to a DF
                results.append((type(model).__name__, len(subset), mod.best_score_, str(mod.best_params_), test_err))

        cv_summary_df = pd.DataFrame(results, columns=['algo_string','subset_index','cv_score','params','test_score'])
        cv_summary_df = cv_summary_df.sort_values(by=['cv_score'], ascending=False).reset_index(drop=True)
        
        return cv_summary_df

        
    def increment_cv(self, train_df, test_df, Y, feature_importance, thresholds,
                       grid_dict=None):
        '''
        Adds more test cases to cv - need to think about this more
        '''
        new_summary_df = self.get_model_cv(train_df, test_df, Y, feature_importance, thresholds)

        self.cv_summary_df = pd.concat([self.cv_summary_df, new_summary_df])
        self.cv_summary_df = self.cv_summary_df.sort_values(by=['cv_score'], ascending=False).reset_index(drop=True)
        
        self.get_best_model(train_df, test_df, Y, feature_importance)

        
    def get_best_model(self, train_df, test_df, Y, feature_importance, thresholds = [0.8, 0.9, 0.95, 0.99],
                       grid_dict=None):
        '''
        Fit a model to the best cv results
        '''
        
        if self.cv_summary_df is None:
            self.cv_summary_df = self.get_model_cv(train_df, test_df, Y, feature_importance, thresholds, grid_dict)
    
        #Get best model
        best_params = ast.literal_eval(self.cv_summary_df.loc[0]['params'])

        best_subset_index = self.cv_summary_df.loc[0]['subset_index']
        self.best_subset = feature_importance.feat_imp_df.x.values[0:best_subset_index]
        
        #This converts the class string back a class, and uses the best parameters found in CV
        self.best_model = eval(self.cv_summary_df.loc[0]['algo_string'])(**best_params) 
        self.best_model.fit(train_df[self.best_subset], train_df[Y])