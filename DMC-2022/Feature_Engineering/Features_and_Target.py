import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors




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
