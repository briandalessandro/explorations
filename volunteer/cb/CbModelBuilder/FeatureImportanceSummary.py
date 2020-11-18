import pandas as pd
import numpy as np
import os
import ast
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import roc_auc_score, precision_score, zero_one_loss, recall_score

class FeatureImportanceSummary(object):
    '''
    Takes in a data frame, features to drop, target_variable
    
        Builds a RF to compute feature importance scores, also does linear regression to get coefficients and signs
    
        Produces a dataFrame with the feature importance summary
        Gives indices within the dataframe at different thresholds of feature importance
    '''
    def __init__(self, df, non_model_features, Y):    
        self.df = df
        self.non_model_features = non_model_features #Features that shouldn't be in a model
        self.Y = Y #Target variables
        self.feat_imp_df = None
        self.feat_imp_thresholds = [0.25, 0.5, 0.8, 0.9, 0.95, 0.99] #thresholds for normalized feature importance
        self.feat_imp_index = {}

    def get_feature_importance_summary(self):

        model_train = self.df.drop(self.non_model_features, axis=1).copy()

        #Fit RF model to get its feature importances
        rf = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=5)
        rf.fit(model_train.drop(self.Y,axis=1), model_train[self.Y])

        #Add to DF
        self.feat_imp_df = pd.DataFrame({'x':model_train.drop(self.Y,axis=1).columns.values,'imp':rf.feature_importances_})
        self.feat_imp_df = self.feat_imp_df.sort_values(by=['imp'], ascending=False).reset_index(drop=True)

        #Get sign of relationship from univariate linear regressions
        xs = self.feat_imp_df.x.values

        betas = []
        signs = []

        for x in xs:
            beta = LinearRegression().fit(model_train[x].values.reshape(-1,1), model_train[self.Y]).coef_[0]
            betas.append(beta)
            if beta < 0:
                signs.append('neg')
            else:
                signs.append('pos')

        self.feat_imp_df['lin_coef'] = betas
        self.feat_imp_df['direction'] = signs
        
        self._get_feature_imp_cutoffs()


    def _get_feature_imp_cutoffs(self):

        cs = np.cumsum(self.feat_imp_df.imp.values)
        
        for cut in self.feat_imp_thresholds:
            self.feat_imp_index[cut] = np.argmin((cs - cut)**2)