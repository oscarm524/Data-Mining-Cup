import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, manhattan_distances

def top_5(book, items, similarity_measure):
    
    """
    
    This function extracts the top-five similar books for a given book and 
    similarity measure. This function takes the following arguments:
    
    book: the book for which five recommendations need to be provided
    
    items: data-frame that contains all the books with their corresponding 
    engineered features.
    
    similarity_measure: possible values Euclidean, Cosine or Manhattan
    
    """
    
    ## Filter out books with same title but different publisher
    temp = items[items['itemID'] == book]
    temp_title = items.loc[items['itemID'] == book, 'title']
    items = items[~np.isin(items['title'], temp_title)]
    items = pd.concat([temp, items]).reset_index(drop = True)
        
    ## Selecting books based on the same language and topic
    items = items[np.isin(items['language'], temp['language'])].reset_index(drop = True)
    
    if (items[np.isin(items['general_topic'], temp['general_topic'])].shape[0] > 5):        
        if (sum(items['general_topic'] == 'Y') > 15000):

            if (all(temp['general_topic_2'] == 'YF') == True):

                items = items[np.isin(items['general_topic_3'], temp['general_topic_3'])].reset_index(drop = True)

            else:
                
                if (items[np.isin(items['general_topic_2'], temp['general_topic_2'])].shape[0] >= 6):
                    
                    items = items[np.isin(items['general_topic_2'], temp['general_topic_2'])].reset_index(drop = True) 
                
                else:
                    
                    items = items[np.isin(items['general_topic'], temp['general_topic'])].reset_index(drop = True)
            
    ## Selecting variables of interest 
    to_remove = ['itemID', 'title', 'author', 'publisher', 'subtopics', 'general_topic', 'general_topic_2', 'general_topic_3', 'language', 'main topic']
    variables_of_interest = items.columns[~np.isin(items.columns, to_remove)]
    items_temp = items[variables_of_interest]
        
    ## Selecting top 5 similar books
    if (similarity_measure == 'Euclidean'):

        D = euclidean_distances(items_temp)
        to_select = np.argsort(D[:, 0])[1:6]
            
    elif (similarity_measure == 'Cosine'):
        
        D = cosine_similarity(items_temp)
        to_select = np.argsort(-D[:, 0])[1:6]
        
    elif (similarity_measure == 'Manhattan'):
        
        D = manhattan_distances(items_temp)
        to_select = np.argsort(D[:, 0])[1:6]
        
    return [items.loc[to_select[0], 'itemID'], items.loc[to_select[1], 'itemID'], items.loc[to_select[2], 'itemID'], items.loc[to_select[3], 'itemID'], items.loc[to_select[4], 'itemID']]
        
     

def top_5_after_transaction(book, book_to_recommend, items, similarity_measure):
    
    """
    
    This function extracts the top-five similar books for a given book, books from 
    transaction history, items and a similarity measure. This function takes the 
    following arguments:
    
    book: the book for which five recommendations need to be provided.
    
    book_to_recommend: list of book from historical transactions.
    
    items: data-frame that contains all the books with their corresponding 
    engineered features.
    
    similarity_measure: possible values Euclidean, Cosine or Manhattan
    
    """
    
    ## Selecting books based on transactions
    items_temp = items.loc[np.isin(items['itemID'], book_to_recommend)]
    
    ## Selecting books based on the same language and topic
    temp = items[items['itemID'] == book]
    temp_title = items.loc[items['itemID'] == book, 'title']
    
    items_temp = items_temp[~np.isin(items_temp['title'], temp_title)]
    items_temp = pd.concat([temp, items_temp]).reset_index(drop = True)
    
    ## Selecting books based on language
    items_temp = items_temp[np.isin(items_temp['language'], temp['language'])].reset_index(drop = True)
    
    ## Selecting variables of interest
    to_remove = ['itemID', 'title', 'author', 'publisher', 'subtopics', 'general_topic', 'general_topic_2', 'general_topic_3', 'language', 'main topic']
    variables_of_interest = items.columns[~np.isin(items.columns, to_remove)]
    items_temp_1 = items_temp[variables_of_interest]
    
    ## Sanity check 
    if (items_temp.shape[0] >= 6):
        
        ## Selecting top 5 similar books
        if (similarity_measure == 'Euclidean'):
            
            D = euclidean_distances(items_temp_1)
            to_select = np.argsort(D[:, 0])[1:6]
        
        elif (similarity_measure == 'Cosine'):
            
            D = cosine_similarity(items_temp_1)
            to_select = np.argsort(-D[:, 0])[1:6]
            
        elif (similarity_measure == 'Manhattan'):
            
            D = manhattan_distances(items_temp_1)
            to_select = np.argsort(D[:, 0])[1:6]
    
        return [items_temp.loc[to_select[0], 'itemID'], items_temp.loc[to_select[1], 'itemID'], items_temp.loc[to_select[2], 'itemID'], items_temp.loc[to_select[3], 'itemID'], items_temp.loc[to_select[4], 'itemID']]

    else:
        
        knn_top_5 = top_5(book, items, similarity_measure)
        return knn_top_5

        
        