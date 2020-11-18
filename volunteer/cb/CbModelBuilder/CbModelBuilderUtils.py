import pandas as pd
import numpy as np
import os
import ast
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import roc_auc_score, precision_score, zero_one_loss, recall_score
from sklearn.utils.multiclass import unique_labels


#These are features we'll exclude, as they're not useful for generalization, and they're specific to the data set used in summer 2019 developing
non_model_features = ['start','submission_time','end','device_id','country','developer_code', 
                      'village_code', 'joint_code', 'gps_coordinates',
                       'gps_coordinates_latitude', 'gps_coordinates_longitude',
                       'gps_coordinates_altitude', 'gps_coordinates_precision',
                       'meter_number','survey_version','meta_instanceid', 'id', 'uuid', 'index', 
                       'parent_index', 'customer_id','start_date','end_date','tariff_min','tariff_max']


def evaluate_regression(predictions, labels):
    errors = abs(predictions - labels)
    print('MAE', mean_absolute_error(predictions, labels))
    print('MSE', mean_squared_error(predictions, labels))
    print('R2', r2_score(labels, predictions))
    print('Average Error: ', np.mean(errors), 'kwh')
    
    
def evaluate_classification(predictions, labels, cutoff=0.5):
    classes = 1*(predictions > cutoff)
    print('AUC', roc_auc_score(labels, predictions))
    print('Accuracy', zero_one_loss(labels, classes))
    print('Precision', precision_score(labels, classes))
    print('Recall: ', recall_score(labels, classes))

def cap_outlier(df, col, max_threshold):
    '''
    Caps the specified column at the max_threshold 
    '''
    df.loc[df[col]>max_threshold, col] = max_threshold
    return df

def train_test_split(df, test_pct, random_state=1):
    '''
    My own train_test_split helper function - keeps target in same DF
    '''
    model_df_shuf = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    #Data is shuffled so we can just take bottom 10% for test set
    test_pct = 0.1
    test_index = int(model_df_shuf.shape[0]) * test_pct
    test_df = model_df_shuf.loc[:test_index]
    train_df = model_df_shuf.loc[test_index:]
    
    return train_df, test_df




def tariff_pdp(df, model, tariff_min=0, tariff_max=95):
    '''
    Inputs:
    df - data frame that can be scored in the model - it should conform to model expectations
    model - the regression that computes expected consumption
    tariff_min - the lowest value of tariff you want to explore, expressed as a percentile
    tariff_max - the highest value of tariff you want to explore, expressed as a percentile
    
    Loops through a range of tariff values to compute E[Consumption|Tariff]
    
    Returns a dataFrame with plotable results

    '''
    
    #Define the range of tariffs to explore
    tariff_range = list(np.percentile(df['tariff'], [tariff_min, tariff_max]))
    tariff_set = np.linspace(tariff_range[0], tariff_range[1], 100)

    pdp = []


    for tar in tariff_set:
        new_df = df.copy()
    
        #Set the tariff to the same level for
        new_df['tariff'] = tar
    
        pdp.append(model.predict(new_df).mean())
        
    return pd.DataFrame({'tariff':tariff_set, 'consumption':pdp})


def segmented_pdp(df, model, segment_var, split_value, tariff_min=0, tariff_max=95):
    '''
    Returns a PDP by population segment
    
    Inputs:
    df - dataframe to be used
    model - model to score
    segment_var - variable to split upon
    split_value - value to split upon
    
    Returns a data frame of PDPs by segment    
    '''
            
    pdp_hi = tariff_pdp(df[(df[segment_var]>=split_value)],model, tariff_min, tariff_max)
    pdp_hi['split'] = 1
    pdp_lo = tariff_pdp(df[(df[segment_var]<split_value)],model, tariff_min, tariff_max)
    pdp_lo['split'] = 0
    
    return pd.concat([pdp_lo, pdp_hi])


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

    