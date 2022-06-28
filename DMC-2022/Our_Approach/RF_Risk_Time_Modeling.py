import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor

########################
## Reading data-files ##
########################

train = pd.read_csv('train_results.csv')
train['last_purchase_date'] = pd.to_datetime(train['last_purchase_date'], format = '%Y-%m-%d')

test = pd.read_csv('test_results_new.csv')
test['last_purchase_date'] = pd.to_datetime(test['last_purchase_date'], format = '%Y-%m-%d')
test['pred_time'] = np.nan

submission = pd.read_csv('submission_results_new.csv')
submission['pred_time'] = np.nan

###########################################
## Engineering quadratic and cubic terms ##
###########################################

train['risk_2'] = train['risk']**2
train['risk_3'] = train['risk']**3

test['risk_2'] = test['risk']**2
test['risk_3'] = test['risk']**3

submission['risk_2'] = submission['risk']**2
submission['risk_3'] = submission['risk']**3

###########################
## Train & Test Datasets ##
###########################

train_less_4 = train[train['risk'] <= 4].reset_index(drop = True)
train_greater_4 = train[train['risk'] > 4].reset_index(drop = True)

test_less_4 = test[test['risk'] <= 4].reset_index(drop = True)
test_greater_4 = test[test['risk'] > 4].reset_index(drop = True)

submission_less_4 = submission[submission['risk'] <= 4].reset_index(drop = True)
submission_greater_4 = submission[submission['risk'] > 4].reset_index(drop = True)

###########################
## Less than 4 Quadratic ##
###########################

X_train_less_4_quadratic = train_less_4[['risk', 'risk_2']]
X_test_less_4_quadratic = test_less_4[['risk', 'risk_2']]
submission_less_4_quadratic = submission_less_4[['risk', 'risk_2']]

Y_train_less_4 = train_less_4['time']
Y_test_less_4 = test_less_4['time']

def objective_less_4_quadratic(trial):
    
    RF_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 300, 50),
                     'min_samples_split': trial.suggest_int('min_samples_split', 5, 19, 2),
                     'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 9, 2),
                     'max_depth' : trial.suggest_int('max_depth', 3, 9, 2)}
    
    model = RandomForestRegressor(**RF_param_grid, n_jobs = -1) 
    
    model.fit(X_train_less_4_quadratic, Y_train_less_4)
        
    ## Predicting on train and test
    RF_pred_test =  model.predict(X_test_less_4_quadratic)
    
    abs_diff = -np.mean(abs(RF_pred_test - Y_test_less_4))
    
    return abs_diff

study_less_4_quadratic = optuna.create_study(study_name = 'Less than 4 quadratic', direction = 'maximize')
study_less_4_quadratic.optimize(objective_less_4_quadratic, n_trials = 100)

study_less_4_quadratic_score = study_less_4_quadratic.best_trial.value

#######################
## Less than 4 Cubic ##
#######################

X_train_less_4_cubic = train_less_4[['risk', 'risk_2', 'risk_3']]
X_test_less_4_cubic = test_less_4[['risk', 'risk_2', 'risk_3']]
submission_less_4_cubic = submission_less_4[['risk', 'risk_2', 'risk_3']]

def objective_less_4_cubic(trial):
    
    RF_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 300, 50),
                     'min_samples_split': trial.suggest_int('min_samples_split', 5, 19, 2),
                     'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 9, 2),
                     'max_depth' : trial.suggest_int('max_depth', 3, 9, 2)}
    
    model = RandomForestRegressor(**RF_param_grid, n_jobs = -1) 
    
    model.fit(X_train_less_4_cubic, Y_train_less_4)
        
    ## Predicting on train and test
    RF_pred_test =  model.predict(X_test_less_4_cubic)
    
    abs_diff = -np.mean(abs(RF_pred_test - Y_test_less_4))
    
    return abs_diff

study_less_4_cubic = optuna.create_study(study_name = 'Less than 4 cubic', direction = 'maximize')
study_less_4_cubic.optimize(objective_less_4_cubic, n_trials = 100)

study_less_4_cubic_score = study_less_4_cubic.best_trial.value

#############################################
## Less than 4: Predicting with best model ##
#############################################

if (study_less_4_cubic_score > study_less_4_quadratic_score):
    
    RF_md = RandomForestRegressor(**study_less_4_cubic.best_params).fit(X_train_less_4_cubic, Y_train_less_4)
    
    test['pred_time'][test['risk'] <= 4] = RF_md.predict(X_test_less_4_cubic)
    submission['pred_time'][submission['risk'] <= 4] = RF_md.predict(submission_less_4_cubic)
    
else: 
    
    RF_md = RandomForestRegressor(**study_less_4_quadratic.best_params).fit(X_train_less_4_quadratic, Y_train_less_4)
    
    test['pred_time'][test['risk'] <= 4] = RF_md.predict(X_test_less_4_quadratic)
    submission['pred_time'][submission['risk'] <= 4] = RF_md.predict(submission_less_4_quadratic)
    
##############################
## Greater than 4 Quadratic ##
##############################

X_train_greater_4_quadratic = train_greater_4[['risk', 'risk_2']]
X_test_greater_4_quadratic = test_greater_4[['risk', 'risk_2']]
submission_greater_4_quadratic = submission_greater_4[['risk', 'risk_2']]

Y_train_greater_4 = train_greater_4['time']
Y_test_greater_4 = test_greater_4['time']

def objective_greater_4_quadratic(trial):
    
    RF_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 300, 50),
                     'min_samples_split': trial.suggest_int('min_samples_split', 5, 19, 2),
                     'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 9, 2),
                     'max_depth' : trial.suggest_int('max_depth', 3, 9, 2)}
    
    model = RandomForestRegressor(**RF_param_grid, n_jobs = -1) 
    
    model.fit(X_train_greater_4_quadratic, Y_train_greater_4)
        
    ## Predicting on train and test
    RF_pred_test =  model.predict(X_test_greater_4_quadratic)
    
    abs_diff = -np.mean(abs(RF_pred_test - Y_test_greater_4))
    
    return abs_diff

study_greater_4_quadratic = optuna.create_study(study_name = 'Greater than 4 quadratic', direction = 'maximize')
study_greater_4_quadratic.optimize(objective_greater_4_quadratic, n_trials = 100)

study_greater_4_quadratic_score = study_greater_4_quadratic.best_trial.value

##########################
## Greater than 4 Cubic ##
##########################

X_train_greater_4_cubic = train_greater_4[['risk', 'risk_2', 'risk_3']]
X_test_greater_4_cubic = test_greater_4[['risk', 'risk_2', 'risk_3']]
submission_greater_4_cubic = submission_greater_4[['risk', 'risk_2', 'risk_3']]

def objective_greater_4_cubic(trial):
    
    RF_param_grid = {'n_estimators': trial.suggest_int('n_estimators', 100, 300, 50),
                     'min_samples_split': trial.suggest_int('min_samples_split', 5, 19, 2),
                     'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 9, 2),
                     'max_depth' : trial.suggest_int('max_depth', 3, 9, 2)}
    
    model = RandomForestRegressor(**RF_param_grid, n_jobs = -1) 
    
    model.fit(X_train_greater_4_cubic, Y_train_greater_4)
        
    ## Predicting on train and test
    RF_pred_test =  model.predict(X_test_greater_4_cubic)
    
    abs_diff = -np.mean(abs(RF_pred_test - Y_test_greater_4))
    
    return abs_diff

study_greater_4_cubic = optuna.create_study(study_name = 'Greater than 4 cubic', direction = 'maximize')
study_greater_4_cubic.optimize(objective_greater_4_cubic, n_trials = 100)

study_greater_4_cubic_score = study_greater_4_cubic.best_trial.value

################################################
## Greater than 4: Predicting with best model ##
################################################

if (study_greater_4_cubic_score > study_greater_4_quadratic_score):
    
    RF_md = RandomForestRegressor(**study_greater_4_cubic.best_params).fit(X_train_greater_4_cubic, Y_train_greater_4)
    
    test['pred_time'][test['risk'] > 4] = RF_md.predict(X_test_greater_4_cubic)
    submission['pred_time'][submission['risk'] > 4] = RF_md.predict(submission_greater_4_cubic)
    
else: 
    
    RF_md = RandomForestRegressor(**study_greater_4_quadratic.best_params).fit(X_train_greater_4_quadratic, Y_train_greater_4)
    
    test['pred_time'][test['risk'] > 4] = RF_md.predict(X_test_greater_4_quadratic)
    submission['pred_time'][submission['risk'] > 4] = RF_md.predict(submission_greater_4_quadratic)


## writing results    
test.to_csv('test_results_pred_time.csv', index = False)
submission.to_csv('submission_results_pred_time.csv', index = False)