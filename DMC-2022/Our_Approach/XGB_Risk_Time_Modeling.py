## Final Modeling with XGBoost Regressor
## Data Mining Cup 2022

## pip install optuna xgboost

## Importing libraries
import boto3
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor

## Defining the bucket
s3 = boto3.resource('s3')
bucket_name = 'evan-callaghan-bucket'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key = 'Data_Mining_Cup_2022/train_results.csv'
file_key2 = 'Data_Mining_Cup_2022/test_results.csv'
file_key3 = 'Data_Mining_Cup_2022/submission_results.csv'

bucket_object = bucket.Object(file_key)
bucket_object2 = bucket.Object(file_key2)
bucket_object3 = bucket.Object(file_key3)

file_object = bucket_object.get()
file_object2 = bucket_object2.get()
file_object3 = bucket_object3.get()

file_content_stream = file_object.get('Body')
file_content_stream2 = file_object2.get('Body')
file_content_stream3 = file_object3.get('Body')

## Reading the data
train = pd.read_csv(file_content_stream)
test = pd.read_csv(file_content_stream2)
submission = pd.read_csv(file_content_stream3)

## Cleaning the data
train['last_purchase_date'] = pd.to_datetime(train['last_purchase_date'], format = '%Y-%m-%d')
test['last_purchase_date'] = pd.to_datetime(test['last_purchase_date'], format = '%Y-%m-%d')

test['pred_time'] = np.nan
submission['pred_time'] = np.nan

## Creating new squared and cubed variables
train['risk_2'] = train['risk']**2
train['risk_3'] = train['risk']**3

test['risk_2'] = test['risk']**2
test['risk_3'] = test['risk']**3

submission['risk_2'] = submission['risk']**2
submission['risk_3'] = submission['risk']**3

## Subsetting the data based on a split in risk score
train_less_4 = train[train['risk'] <= 4].reset_index(drop = True)
train_greater_4 = train[train['risk'] > 4].reset_index(drop = True)

test_less_4 = test[test['risk'] <= 4].reset_index(drop = True)
test_greater_4 = test[test['risk'] > 4].reset_index(drop = True)

submission_less_4 = submission[submission['risk'] <= 4].reset_index(drop = True)
submission_greater_4 = submission[submission['risk'] > 4].reset_index(drop = True)


############################################
## Modeling: Quadratic model w/ risk <= 4 ##
############################################

## Printing indicator for the terminal window
print('\n Starting: Quadratic model w/ risk <= 4 \n')

def objective_less_quadratic(trial):
    
    ## Defining the XGB hyper-parameter grid
    XGB_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 700, 100),
                     'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.951, step = 0.05),
                     'min_split_loss': trial.suggest_int('min_split_loss', 0, 5, 1),
                     'max_depth' : trial.suggest_int('max_depth', 3, 7, 1),
                     'min_child_weight' : trial.suggest_int('min_child_weight', 5, 9, 1),
                     'subsample' : trial.suggest_float('subsample', 0.6, 1, step = 0.1),
                     'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1, step = 0.1)}
    
    ## Building the XGBRegressor model
    model = XGBRegressor(**XGB_param_grid, n_jobs = -1).fit(X_train_less_quadratic, Y_train_less_quadratic)
        
    ## Predicting on the test data-frame
    XGB_pred_test = model.predict(X_test_less_quadratic)
    
    ## Evaluating model performance on the test set
    abs_diff = -np.mean(abs(XGB_pred_test - Y_test_less_quadratic))
    
    ## Returning absolute difference of model test predictions
    return abs_diff

## Defining input and target variables
X_train_less_quadratic = train_less_4[['risk', 'risk_2']]
Y_train_less_quadratic = train_less_4['time']

X_test_less_quadratic = test_less_4[['risk', 'risk_2']]
Y_test_less_quadratic = test_less_4['time']

submission_less_quadratic = submission_less_4[['risk', 'risk_2']]

## Calling Optuna objective function
study_less_quadratic = optuna.create_study(direction = 'maximize')
study_less_quadratic.optimize(objective_less_quadratic, n_trials = 100)

## Storing best performance value
best_score_less_quadratic = study_less_quadratic.best_trial.value
print('Best Score:', best_score_less_quadratic)


########################################
## Modeling: Cubic model w/ risk <= 4 ##
########################################

## Printing indicator for the terminal window
print('\n Starting: Cubic model w/ risk <= 4 \n')

def objective_less_cubic(trial):
    
    ## Defining the XGB hyper-parameter grid
    XGB_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 700, 100),
                     'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.951, step = 0.05),
                     'min_split_loss': trial.suggest_int('min_split_loss', 0, 5, 1),
                     'max_depth' : trial.suggest_int('max_depth', 3, 7, 1),
                     'min_child_weight' : trial.suggest_int('min_child_weight', 5, 9, 1),
                     'subsample' : trial.suggest_float('subsample', 0.6, 1, step = 0.1),
                     'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1, step = 0.1)}
    
    ## Building the XGBRegressor model
    model = XGBRegressor(**XGB_param_grid, n_jobs = -1).fit(X_train_less_cubic, Y_train_less_cubic)
        
    ## Predicting on the test data-frame
    XGB_pred_test = model.predict(X_test_less_cubic)
    
    ## Evaluating model performance on the test set
    abs_diff = -np.mean(abs(XGB_pred_test - Y_test_less_cubic))
    
    ## Returning absolute difference of model test predictions
    return abs_diff

## Defining input and target variables
X_train_less_cubic = train_less_4[['risk', 'risk_2', 'risk_3']]
Y_train_less_cubic = train_less_4['time']

X_test_less_cubic = test_less_4[['risk', 'risk_2', 'risk_3']]
Y_test_less_cubic = test_less_4['time']

submission_less_cubic = submission_less_4[['risk', 'risk_2', 'risk_3']]

## Calling Optuna objective function
study_less_cubic = optuna.create_study(direction = 'maximize')
study_less_cubic.optimize(objective_less_cubic, n_trials = 100)

## Storing best performance value
best_score_less_cubic = study_less_cubic.best_trial.value
print('Best Score:', best_score_less_cubic)


#########################################
## Predicting: Best model w/ risk <= 4 ##
#########################################

## Fitting a new model with the optimal set of hyper-parameters
if (best_score_less_quadratic > best_score_less_cubic):
    
    ## Building the model
    xgb_md = XGBRegressor(**study_less_quadratic.best_params).fit(X_train_less_quadratic, Y_train_less_quadratic)
    
    ## Predicting on the test set
    test['pred_time'][test['risk'] <= 4] = xgb_md.predict(X_test_less_quadratic)
    
    ## Predicting on the submission set
    submission['pred_time'][submission['risk'] <= 4] = xgb_md.predict(submission_less_quadratic)
    
else:
    
    ## Building the model
    xgb_md = XGBRegressor(**study_less_cubic.best_params).fit(X_train_less_cubic, Y_train_less_cubic)
    
    ## Predicting on the test set
    test['pred_time'][test['risk'] <= 4] = xgb_md.predict(X_test_less_cubic)
    
    ## Predicting on the submission set
    submission['pred_time'][submission['risk'] <= 4] = xgb_md.predict(submission_less_cubic)
    

############################################
## Modeling: Quadratic model w/ risk > 4 ##
############################################

## Printing indicator for the terminal window
print('\n Starting: Quadratic model w/ risk > 4 \n')

def objective_greater_quadratic(trial):
    
    ## Defining the XGB hyper-parameter grid
    XGB_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 700, 100),
                     'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.951, step = 0.05),
                     'min_split_loss': trial.suggest_int('min_split_loss', 0, 5, 1),
                     'max_depth' : trial.suggest_int('max_depth', 3, 7, 1),
                     'min_child_weight' : trial.suggest_int('min_child_weight', 5, 9, 1),
                     'subsample' : trial.suggest_float('subsample', 0.6, 1, step = 0.1),
                     'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1, step = 0.1)}
    
    ## Building the XGBRegressor model
    model = XGBRegressor(**XGB_param_grid, n_jobs = -1).fit(X_train_greater_quadratic, Y_train_greater_quadratic)
        
    ## Predicting on the test data-frame
    XGB_pred_test = model.predict(X_test_greater_quadratic)
    
    ## Evaluating model performance on the test set
    abs_diff = -np.mean(abs(XGB_pred_test - Y_test_greater_quadratic))
    
    ## Returning absolute difference of model test predictions
    return abs_diff

## Defining input and target variables
X_train_greater_quadratic = train_greater_4[['risk', 'risk_2']]
Y_train_greater_quadratic = train_greater_4['time']

X_test_greater_quadratic = test_greater_4[['risk', 'risk_2']]
Y_test_greater_quadratic = test_greater_4['time']

submission_greater_quadratic = submission_greater_4[['risk', 'risk_2']]

## Calling Optuna objective function
study_greater_quadratic = optuna.create_study(direction = 'maximize')
study_greater_quadratic.optimize(objective_greater_quadratic, n_trials = 100)

## Storing best performance value
best_score_greater_quadratic = study_greater_quadratic.best_trial.value
print('Best Score:', best_score_greater_quadratic)


########################################
## Modeling: Cubic model w/ risk > 4 ##
########################################

## Printing indicator for the terminal window
print('\n Starting: Cubic model w/ risk > 4 \n')

def objective_greater_cubic(trial):
    
    ## Defining the XGB hyper-parameter grid
    XGB_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 700, 100),
                     'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.951, step = 0.05),
                     'min_split_loss': trial.suggest_int('min_split_loss', 0, 5, 1),
                     'max_depth' : trial.suggest_int('max_depth', 3, 7, 1),
                     'min_child_weight' : trial.suggest_int('min_child_weight', 5, 9, 1),
                     'subsample' : trial.suggest_float('subsample', 0.6, 1, step = 0.1),
                     'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1, step = 0.1)}
    
    ## Building the XGBRegressor model
    model = XGBRegressor(**XGB_param_grid, n_jobs = -1).fit(X_train_greater_cubic, Y_train_greater_cubic)
        
    ## Predicting on the test data-frame
    XGB_pred_test = model.predict(X_test_greater_cubic)
    
    ## Evaluating model performance on the test set
    abs_diff = -np.mean(abs(XGB_pred_test - Y_test_greater_cubic))
    
    ## Returning absolute difference of model test predictions
    return abs_diff

## Defining input and target variables
X_train_greater_cubic = train_greater_4[['risk', 'risk_2', 'risk_3']]
Y_train_greater_cubic = train_greater_4['time']

X_test_greater_cubic = test_greater_4[['risk', 'risk_2', 'risk_3']]
Y_test_greater_cubic = test_greater_4['time']

submission_greater_cubic = submission_greater_4[['risk', 'risk_2', 'risk_3']]

## Calling Optuna objective function
study_greater_cubic = optuna.create_study(direction = 'maximize')
study_greater_cubic.optimize(objective_greater_cubic, n_trials = 100)

## Storing best performance value
best_score_greater_cubic = study_greater_cubic.best_trial.value
print('Best Score:', best_score_greater_cubic)


#########################################
## Predicting: Best model w/ risk > 4 ##
#########################################

## Fitting a new model with the optimal set of hyper-parameters
if (best_score_greater_quadratic > best_score_greater_cubic):
    
    ## Building the model
    xgb_md = XGBRegressor(**study_greater_quadratic.best_params).fit(X_train_greater_quadratic, Y_train_greater_quadratic)
    
    ## Predicting on the test set
    test['pred_time'][test['risk'] > 4] = xgb_md.predict(X_test_greater_quadratic)
    
    ## Predicting on the submission set
    submission['pred_time'][submission['risk'] > 4] = xgb_md.predict(submission_greater_quadratic)
    
else:
    
    ## Building the model
    xgb_md = XGBRegressor(**study_greater_cubic.best_params).fit(X_train_greater_cubic, Y_train_greater_cubic)
    
    ## Predicting on the test set
    test['pred_time'][test['risk'] > 4] = xgb_md.predict(X_test_greater_cubic)
    
    ## Predicting on the submission set
    submission['pred_time'][submission['risk'] > 4] = xgb_md.predict(submission_greater_cubic)


#############
## Results ##
#############

## Exporting the test and submission data-frames as csv files 
test.to_csv('test_results_pred_time_XGB.csv', index = False)
submission.to_csv('submission_results_pred_time_XGB.csv', index = False)