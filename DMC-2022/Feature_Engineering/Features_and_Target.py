import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


def Surv_Features_1(train, test):
    
    ## Defingin train_ids and test_ids
    train_ids = train['user_item'].unique()
    print('The number of unique user-item in train is:', len(train_ids))
    test_ids = test['user_item'].unique()
    print('The number of unique user-item in test is:', len(test_ids))
    
    ## Defining data-frame to be exported
    data_out = pd.DataFrame({'user_item': train_ids})
    data_out['tot_purchases'] = np.nan
    data_out['Frequency'] = np.nan
    data_out['weeks_to_magic_date'] = np.nan
    data_out['target_1'] = np.nan
    data_out['target_2'] = np.nan
    
    n = data_out.shape[0]
    
    for i in tqdm(range(0, n)):
        
        ## Subsetting data based on user_item
        temp = train[train['user_item'] == train_ids[i]].reset_index(drop = True)
        
        ## Computing frequency
        if (temp.shape[0] ==  1):
        
            data_out.loc[i, 'Frequency'] = 31 
        
        else: 
            
            data_out.loc[i, 'Frequency'] = ((pd.to_datetime('2021-01-04') - temp['date'].iloc[0]) / np.timedelta64(1, 'W')) / temp.shape[0]
        
        ## Extracting total purchases
        data_out.loc[i, 'tot_purchases'] = temp.shape[0]
        
        ## Computing weeks to magic date
        data_out.loc[i, 'weeks_to_magic_date'] = (pd.to_datetime('2021-01-04') - temp['date'].iloc[-1]) / np.timedelta64(1, 'W')
    
        ## Checking if the user_item appear in test
        to_check = np.isin(train_ids[i], test_ids)
        
        if (to_check):
            
            temp_test = test[test['user_item'] == train_ids[i]]
            date_to_check = pd.to_datetime(temp_test['date'].iloc[0]) 
            
            data_out.loc[i, 'target_1'] = 1
            data_out.loc[i, 'target_2'] = (date_to_check - temp['date'].iloc[-1]) / np.timedelta64(1, 'W')
                
        else: 
            
            data_out.loc[i, 'target_1'] = 0
            data_out.loc[i, 'target_2'] = (pd.to_datetime('2021-01-31') - temp['date'].iloc[-1]) / np.timedelta64(1, 'W')
            
    
    data_out['target_1'] = data_out['target_1'].apply(int)
    data_out['target_2'] = data_out['target_2'].apply(int)
    x = data_out['user_item'].str.split('_', expand = True)
    data_out['itemID'] = x[1]
    data_out['itemID'] = data_out['itemID'].apply(int)
    
    return data_out 



def Surv_Features_1_submission(train):
    
    ## Defingin train_ids and test_ids
    train_ids = train['user_item'].unique()
    print('The number of unique user-item in train is:', len(train_ids))
    
    ## Defining data-frame to be exported
    data_out = pd.DataFrame({'user_item': train_ids})
    data_out['tot_purchases'] = np.nan
    data_out['Frequency'] = np.nan
    data_out['weeks_to_magic_date'] = np.nan
    
    n = data_out.shape[0]
    
    for i in tqdm(range(0, n)):
        
        ## Subsetting data based on user_item
        temp = train[train['user_item'] == train_ids[i]].reset_index(drop = True)
        
        ## Computing frequency
        if (temp.shape[0] ==  1):
        
            data_out.loc[i, 'Frequency'] = 31 
        
        else: 
            
            data_out.loc[i, 'Frequency'] = ((pd.to_datetime('2022-02-01') - temp['date'].iloc[0]) / np.timedelta64(1, 'W')) / temp.shape[0]
        
        ## Extracting total purchases
        data_out.loc[i, 'tot_purchases'] = temp.shape[0]
        
        ## Computing weeks to magic date
        data_out.loc[i, 'weeks_to_magic_date'] = (pd.to_datetime('2022-02-01') - temp['date'].iloc[-1]) / np.timedelta64(1, 'W')            
    
    x = data_out['user_item'].str.split('_', expand = True)
    data_out['itemID'] = x[1]
    data_out['itemID'] = data_out['itemID'].apply(int)
    
    return data_out 



def score_survival_model(model, X_train, Y_train):
    
    prediction = model.predict(X_train)
    result = concordance_index_censored(Y_train['cens'], Y_train['time'], prediction)
    
    return result[0]



def Features_2(train):
    
    ## Identifying all unique itemID
    items = train['itemID'].unique()
    n = len(items)
    
    ## Defining data-frame to be exported
    data_out = pd.DataFrame({'itemID': items})
    data_out['Avg_Frequency'] = np.nan
    
    for i in range(0, n):
    
        temp = train[train['itemID'] == items[i]]
        temp = pd.DataFrame(temp['userID'].value_counts())
        
        temp['tot_weeks'] = 31
        temp['Frequency'] = temp['tot_weeks'] / temp['userID']
        temp.columns = ['orders', 'tot_weeks', 'Frequency']
        temp = temp[temp['orders'] > 1].reset_index(drop = True)
        
        if (temp.shape[0] > 0):
        
            data_out.loc[i, 'Avg_Frequency'] = temp['Frequency'].mean()
            
        else:
            
            data_out.loc[i, 'Avg_Frequency'] = 31
        
    return data_out
