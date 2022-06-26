import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, manhattan_distances


def top_5(book, items, items_ratings, similarity_measure):
    
    """
    
    This function extracts the top-five similar books for a given book and 
    similarity measure. This function takes the following arguments:
    
    book: the book for which five recommendations need to be provided
    
    items: data-frame that contains all the books with their corresponding engineered features.
    
    items_ratings: data-frame that contains the popularity score of itemIDs with 
    transactional history.
    
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
        
    ## Selecting top similar books
    if (similarity_measure == 'Euclidean'):
    
        D = euclidean_distances(items_temp)
        to_select = np.argsort(D[:, 0])[1:21]
        
    elif (similarity_measure == 'Cosine'):
        
        D = cosine_similarity(items_temp)
        to_select = np.argsort(-D[:, 0])[1:21]

    elif (similarity_measure == 'Manhattan'):
        
        D = manhattan_distances(items_temp)
        to_select = np.argsort(D[:, 0])[1:21]
    
    return items_and_ratings(to_select, items, items_ratings)


def top_5_after_transaction(book, book_to_recommend, items, items_ratings, similarity_measure):
    
    """
    
    This function extracts the top-five similar books for a given book, books from 
    transaction history, items, items popularity score, and a similarity measure. 
    This function takes the following arguments:
    
    book: the book for which five recommendations need to be provided
    
    book_to_recommend: list of book from historical transactions.
    
    items: data-frame that contains all the books with their corresponding engineered features.
    
    items_ratings: data-frame that contains the popularity score of itemIDs with 
    transactional history.
    
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
    
    if (items_temp.shape[0] >= 20):
    
        ## Selecting top 20 similar books and their corresponding ratings
        if (similarity_measure == 'Euclidean'):
        
            D = euclidean_distances(items_temp_1)
            to_select = np.argsort(D[:, 0])[1:21]
            
        elif (similarity_measure == 'Cosine'):
            
            D = cosine_similarity(items_temp_1)
            to_select = np.argsort(-D[:, 0])[1:21]
            
        elif (similarity_measure == 'Manhattan'):
            
            D = manhattan_distances(items_temp_1)
            to_select = np.argsort(D[:, 0])[1:21]
        
        return items_and_ratings(to_select, items_temp, items_ratings)            

    else:

        knn_top_5 = top_5(book, items, items_ratings, similarity_measure)
        
        return knn_top_5
    

def items_and_ratings(to_select, items, ratings):
    
    items_and_ratings = items.loc[to_select, :].reset_index(drop = True)
    items_and_ratings = pd.merge(items_and_ratings, ratings, on = ['itemID'], how = 'left')
        
    items_and_ratings_sort = items_and_ratings.sort_values(by = ['rating'], ascending = False)
    items_and_ratings_sort = items_and_ratings_sort[items_and_ratings_sort['rating'] >= 1].reset_index(drop = True)
        
    if (items_and_ratings_sort.shape[0] >= 5):

        return [items_and_ratings_sort.loc[0, 'itemID'], items_and_ratings_sort.loc[1, 'itemID'], items_and_ratings_sort.loc[2, 'itemID'], items_and_ratings_sort.loc[3, 'itemID'], items_and_ratings_sort.loc[4, 'itemID']]
        
    elif (items_and_ratings_sort.shape[0] >= 1):

        temp_itemID_list = [itemID for itemID in items_and_ratings_sort['itemID']]
        n = len(temp_itemID_list)
        items_and_ratings = items_and_ratings[~np.isin(items_and_ratings['itemID'], temp_itemID_list)].reset_index(drop = True)
        
        to_append = items_and_ratings.loc[0:(4-n), 'itemID'].reset_index(drop = True).values.tolist()
            
        return temp_itemID_list + to_append
        
    else:

        return [items_and_ratings.loc[0, 'itemID'], items_and_ratings.loc[1, 'itemID'], items_and_ratings.loc[2, 'itemID'], items_and_ratings.loc[3, 'itemID'], items_and_ratings.loc[4, 'itemID']]