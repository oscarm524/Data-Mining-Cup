import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest


def extracting_times(Y):
    
    times = list()
    for i in range(0, Y.shape[0]):
        times.append(Y[i][1])
    
    return times


########################
## Reading data-files ##
########################

surv_data = pd.read_csv('survival_modeling_full_data_full_features_date.csv')
submission = pd.read_csv('submission_with_features.csv')
X_submission = submission[['userID', 'itemID', 'user_item', 'tot_purchases', 'Frequency', 'weeks_to_magic_date', 'Avg_Frequency', 'Avg_5_neighbors_freq', 'Avg_10_neighbors_freq', 'Avg_15_neighbors_freq', 'Avg_20_neighbors_freq', 'Avg_25_neighbors_freq', 'Avg_30_neighbors_freq']]


###############################
## Defining input and target ##
###############################

X = surv_data[['user_item', 'tot_purchases', 'Frequency', 'weeks_to_magic_date', 'Avg_Frequency', 'Avg_5_neighbors_freq', 'Avg_10_neighbors_freq', 'Avg_15_neighbors_freq', 'Avg_20_neighbors_freq', 'Avg_25_neighbors_freq', 'Avg_30_neighbors_freq', 'last_purchase_date']]
Y = surv_data[['target_1', 'target_2']]
Y.columns = ['cens', 'time']
Y.loc[:, 'cens'] = Y.loc[:, 'cens'].map({0: False, 1: True})


########################
## Splitting the data ##
########################

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

Y_train = Y_train.to_numpy()
Y_train = [(e1,e2) for e1,e2 in Y_train]
Y_train = np.array(Y_train, dtype = [('cens', '?'), ('time', '<f8')])

Y_test = Y_test.to_numpy()
Y_test = [(e1,e2) for e1,e2 in Y_test]
Y_test = np.array(Y_test, dtype = [('cens', '?'), ('time', '<f8')])

## Fitting Random Forest Survival model 
RF_md = RandomSurvivalForest(n_estimators = 200, min_samples_split = 11, min_samples_leaf = 7, max_depth = 3, n_jobs = -1).fit(X_train.drop(columns = ['user_item', 'tot_purchases', 'last_purchase_date'], axis = 1), Y_train) 


##########################################
## Predting on train, test & submission ##
##########################################

## Extracting times 
times_train = extracting_times(Y_train)
times_test = extracting_times(Y_test)
    
## Predicting on train, test & submission
RF_pred_train = RF_md.predict(X_train.drop(columns = ['user_item', 'tot_purchases', 'last_purchase_date'], axis = 1))
RF_pred_test =  RF_md.predict(X_test.drop(columns = ['user_item', 'tot_purchases', 'last_purchase_date'], axis = 1))
RF_pred_submission = RF_md.predict(X_submission.drop(columns = ['userID', 'itemID', 'user_item', 'tot_purchases'], axis = 1))

##################################
## Saving Results to flat files ##
##################################

X_train['risk'] = RF_pred_train
X_train['time'] = times_train
X_train.to_csv('train_results.csv', index = False)

X_test['risk'] = RF_pred_test
X_test['time'] = times_test
X_test.to_csv('test_results.csv', index = False)

X_submission['risk'] = RF_pred_submission
X_submission.to_csv('submission_results.csv', index = False)