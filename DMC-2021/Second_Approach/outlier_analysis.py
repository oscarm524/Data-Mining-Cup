import pandas as pd
import numpy as np
import scipy.linalg as la
from sklearn.preprocessing import MinMaxScaler


def distance_from_centroid_scores(data):
    
    """
    
    This function computes the outlier scores of the items that haves historical 
    transactions. We used the algorithm presented in page 78 of the Outlier 
    Analysis. Notice that these scores are computed using click, basket and order 
    counts.
    
    """
    
    ## Storing full-dataset
    full_data = data[['itemID', 'click_count', 'basket_count', 'order']]
    
    ## Selecting non-missing data 
    temp_data = data.dropna()
    temp_data = temp_data[['click_count', 'basket_count', 'order']]
    
    ## Standardizing the data
    scaler = MinMaxScaler().fit(temp_data)
    temp_data = scaler.transform(temp_data)

    ## Computing the covariance matrix
    sigma = np.cov(temp_data, rowvar = False)
    
    ## Computing eigenvalues and eigenvectos of the covariance matrix
    eigvals, eigvecs = la.eig(sigma)
    
    ## Defining D and P (for PCA outlier score algorithm form Outlier 
    ## Analysis book)
    D = temp_data
    P = eigvecs

    ## Computing D'
    D_prime = np.matmul(D, P)

    ## Standardizing (dividing each column by it standard deviation)
    for i in range(0, D_prime.shape[1]):
        
        D_prime[:, i] = D_prime[:, i] / D_prime[:, i].std(ddof = 1)
    
    ## Computing the centroid
    centroid = D_prime.mean(axis = 0)
    
    ## Declaring list to store Euclidean distances
    distances = []
    
    ## Finding the number of rows in data
    n = D_prime.shape[0]
    
    for i in range(0, n):
        
        ## Selecting the i-th row
        temp = D_prime[i, :]
        
        ## Computing the Euclidean distance
        distances.append(np.sqrt(np.sum((temp - centroid)**2)))
    
    ## Changing the outlier-scores to 1-5 scale
    scores = changing_scale(distances, low_bound = 1, up_bound = 5)
    
    ## Appending ratings to itemID that appear on transaction file
    temp_data = data.dropna()
    itemIDs = temp_data['itemID']
    temp_data = pd.DataFrame({'itemID': itemIDs})
    temp_data['rating'] = scores
    
    ## Appending ratings to the full-dataset
    data_out = pd.merge(full_data, temp_data, on = ['itemID'], how = 'left')
    data_out['rating'] = data_out['rating'].fillna(0)
    
    return data_out


def changing_scale(data, low_bound, up_bound):
    
    ## Finding the max and the min 
    max_val = max(data)
    min_val = min(data)
    
    ## Chaning the scale of data
    newdata = ((up_bound - low_bound) / (max_val - min_val))*(data - max_val) + up_bound
    
    return newdata