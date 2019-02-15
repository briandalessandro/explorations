import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics  import mutual_info_score

from msft_functions import *

def main():
    
    rel_thresh = 0.001
    n_thresh = 2000
    bias_thresh = 0.6
    feat_imp_thresh = 0.999
    run_rf = False
    run_gbdt = True
    
    modfile_pre = '/Users/briand/data/MSFT_Best_Model_ALL_pre4.pickle'
    modfile = '/Users/briand/data/MSFT_Best_Model_ALL4.pickle'
    
    ################################################################
    #prep initial data
    ################################################################
    datadir = '~/data/'
    train_dev = datadir + 'train_dev.csv'
    train = datadir + 'train_dev.csv'
    
    train_df = pd.read_csv(train)
    train_df.set_index('MachineIdentifier', inplace=True)

    #Filter based on P(TestSet|X)
    bias_df = pd.read_csv(datadir + 'train_bias_weights.csv')
    bias_df.set_index('MachineIdentifier', inplace=True)
    bias_df = bias_df[(bias_df.prob>bias_thresh)]

    train_df = train_df.merge(bias_df, how='inner', left_index=True, right_index=True)
    train_df = train_df.drop('prob', axis=1)
    print('raw data shape')
    print(train_df.shape)
    
    y = 'HasDetections'

    #Organize columns into groups by type
    x_numeric = list(train_df.describe().columns.values)
    x_ident = [x for x in x_numeric if 'Identifier' in x] #Some numeric columns actually IDs
    x_numeric = list(set(x_numeric) - set(x_ident)) #Keep only true numerics
    x_str = list(set(train_df.columns.values) - set(x_numeric))
    x_numeric.remove(y)
    
    train_all_x, train_y, str_trans_train, fillna_train = build_train_data(train_df, 
                                                                           x_numeric, x_str, y,
                                                                           rel_thresh, n_thresh)
    print('feature train shape')
    print(train_all_x.shape)
    
    train_df = None #Clear memory
    
    ################################################################
    #Get robust features
    ################################################################
    
    parts = 4
    def hash_index(row):
        return hash(row.name) % parts

    index_partitions = train_all_x.apply(hash_index, axis=1)

    feat_imp_dict = {}
    for i in range(parts):
        print(i)
        df_x = train_all_x[(index_partitions == i)]
        df_y = train_y[(index_partitions == i)]
        feat_imp_dict[i] = get_rf_featimp(df_x, df_y)
    
    best_featset = get_robust_featimp(train_all_x.columns.values, feat_imp_dict, feat_imp_thresh)
    print('final selected robust feature count')
    print(len(best_featset))
    
    ################################################################
    #Model selection
    ################################################################
    
    parts = 100
    def hash_index(row):
        return hash(row.name) % parts

    index_partitions = train_all_x.apply(hash_index, axis=1)

    holdout_n = min(200000, round(train_all_x.shape[0]*.1))
    partition_n = int(np.floor(parts * (holdout_n / train_all_x.shape[0])))
    part_1 = parts - partition_n

    filt_train = (index_partitions < part_1)
    filt_val = (index_partitions >= part_1)

    trainsamp_x = train_all_x[filt_train]
    trainsamp_y = train_y[filt_train]

    valsamp_x = train_all_x[filt_val]
    valsamp_y = train_y[filt_val]
    
    pg_rf_d = {'n_estimators':[200, 500]}

    pg_gbdt_d = {'n_estimators':[100, 200],
           'learning_rate':[0.05, 0.1],
           'max_depth':[7]}

    aucs = []

    if run_rf:
        pg_rf = ParameterGrid(pg_rf_d)
        for g in pg_rf:
            print(g)
            rf = RandomForestClassifier(**g)
            rf.fit(trainsamp_x[best_featset], trainsamp_y)
            pred = rf.predict_proba(valsamp_x[best_featset])[:,1]
            aucs.append((roc_auc_score(valsamp_y.values,pred), 'rf',g))
            rf = None

    if run_gbdt:
        pg_gbdt = ParameterGrid(pg_gbdt_d)
        for g in pg_gbdt:
            print(g)
            gb = GradientBoostingClassifier(**g)
            gb.fit(trainsamp_x[best_featset], trainsamp_y)
            pred = gb.predict_proba(valsamp_x[best_featset])[:,1]
            aucs.append((roc_auc_score(valsamp_y.values,pred), 'gb',g))
            gp = None
    
    trainsamp_x = None; trainsamp_y = None; valsamp_x = None; valsamp_y = None

    aucs.sort(reverse=True)
    best_auc, best_algo, best_params = aucs[0]
    print('grid search results')
    print(aucs)
    
    ################################################################
    #Train official model
    ################################################################
    
    print(best_algo)
    print(best_params)

    bm_pack_pre = BestModelPackage(None, best_featset, fillna_train, str_trans_train)
        
    with open(modfile_pre, 'wb') as w:
        pickle.dump(bm_pack_pre, w)

    if best_algo=='rf':
        best_model = RandomForestClassifier(**best_params)
    else:
        best_model = GradientBoostingClassifier(**best_params)
    
    best_model.fit(train_all_x[best_featset], train_y)
        
    best_mod = BestModelPackage(best_model, best_featset, fillna_train, str_trans_train)
        
    with open(modfile, 'wb') as w:
        pickle.dump(best_mod, w)

    ################################################################
    #Score test set
    ################################################################

    test_all_x = build_test_data(datadir + 'test_dev.csv', 
                             best_mod.fillna_values, 
                             best_mod.str_transforms)

    test_preds = best_mod.model.predict_proba(test_all_x[best_mod.features])[:,1]

    test_pred_df = pd.DataFrame(test_preds, index=test_all_x.index.values, 
                            columns=['HasDetections'])

    test_pred_df.to_csv(modfile.split('.')[0] + '.csv', sep=',', 
                    header=True, index=True, index_label='MachineIdentifier')
    
    
if __name__=='__main__':
    main()