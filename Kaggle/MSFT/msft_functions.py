import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics  import mutual_info_score

#Helper functions

y = 'HasDetections'

def cond_join(c):
    if len(c[1])==0:
        return c[0]
    else:
        return '{}_{}'.format(c[0],c[1])
    
def entropy(p):
    return -1*(p*np.log(p) + (1-p)*np.log((1-p)))

#base_rate_entropy = entropy(train_df[y].mean())

#def rel_row_entropy(row):
#    e1 = entropy(row.HasDetections['mean'])
#    return (base_rate_entropy - e1) / base_rate_entropy

def row_entropy(row):
    return entropy(row.HasDetections['mean'])

#feature transformation functions

def clean_numeric(df_numeric, filler=None):
    #Clean missing values in numeric data
    #df_numeric = df[x_numeric]

    if filler is None:
        medians = df_numeric.median()
    else:
        medians = filler
    
    df_numeric.fillna(medians, inplace=True)
    return df_numeric, medians

def get_booleans(df, y, x, rel_thresh=0.001, n_thresh=2000):
    '''
    Feature construction/selection:
    - compute entropy in each category. 
    - If relative entropy is above a threshold, convert to binary
    - check to make sure no more than K-1 features

    '''
    base_rate_entropy = entropy(df[y].mean())
    
    df_grp_x = df[[x,y]].groupby(x).agg([len, np.mean]).reset_index()
    df_grp_x['relent'] = df_grp_x.apply(row_entropy, axis=1)
    df_grp_x['relent'] = (base_rate_entropy - df_grp_x['relent']) / base_rate_entropy   
    df_grp_x.columns = [cond_join(c) for c in df_grp_x.columns.values]
    df_grp_x_filt = df_grp_x[(df_grp_x.relent>rel_thresh) & (df_grp_x.HasDetections_len>n_thresh)]
    n_cats = df_grp_x.shape[0]
    chosen = list(df_grp_x_filt[x].values)
    if n_cats == len(chosen):
        return chosen[:-1]
    else:
        return chosen
    
def get_booleans_all_x(df, x_str, rel_thresh=0.001, n_thresh=2000):
    lab = 'MachineIdentifier'
    str_trans = {}

    for x in x_str:
        if x != lab:
            bools = get_booleans(df, y, x, rel_thresh, n_thresh)
            loc_dict = {}
            for i, b in enumerate(bools):
                loc_dict[i] = b        
            str_trans[x] = loc_dict
    return str_trans

def transform_strings(df, str_trans):
    #Now create the string matrix
    df_str = pd.DataFrame(index=df.index)
    
    
    for x in str_trans:
        if len(x) > 0:
            loc_dict = str_trans[x]
            for i in range(len(loc_dict)):
                df_str[x+'_{}'.format(i)] = 1*(df[x]==loc_dict[i])
    return df_str
        
#Training 
def build_train_data(train_df, x_numeric, x_str, y, rel_thresh=0.001, n_thresh=2000):
    str_trans_train = get_booleans_all_x(train_df, x_str, rel_thresh, n_thresh)
    train_df_str = transform_strings(train_df[x_str], str_trans_train)
    train_df_num, fillna_train = clean_numeric(train_df[x_numeric])
    train_all_x = train_df_num.merge(train_df_str, how='inner', left_index=True, right_index=True)
    train_y = train_df[y]
    return train_all_x, train_y, str_trans_train, fillna_train
    
#Test
def build_test_data(test_file, fillna_train, str_trans_train):
    test_df = pd.read_csv(test_file)
    test_df.set_index('MachineIdentifier', inplace=True)
    test_df_str = transform_strings(test_df, str_trans_train)
    test_df_num, fillna_train = clean_numeric(test_df, fillna_train)
    test_all_x = test_df_num.merge(test_df_str, how='inner', left_index=True, right_index=True)
    return test_all_x

class BestModelPackage(object):
    
    def __init__(self, model, features, fillna_values, str_transforms):
        self.model = model
        self.features = features
        self.fillna_values = fillna_values
        self.str_transforms = str_transforms
        
 

def get_rf_featimp(df_x, df_y):
    rf = RandomForestClassifier(n_estimators=100).fit(df_x, df_y)
    return rf.feature_importances_

def get_robust_featimp(x_columns, feat_importances, mi_thresh=0.98):
    
    for i in range(len(feat_importances)):
        fi_sort = feat_importances[i][np.argsort(feat_importances[i])][::-1]
        keep_n = (np.cumsum(fi_sort)<mi_thresh).sum()
        keep_indices = np.argsort(feat_importances[i])[-keep_n:]
        best_feats = x_columns[keep_indices]
        
        if i==0:
            best_set = set(best_feats)
        else:
            best_set = best_set & set(best_feats)
            
    return list(best_set)
