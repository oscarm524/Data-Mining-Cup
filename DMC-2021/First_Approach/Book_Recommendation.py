import pandas as pd
import numpy as np
import kNN

def Book_Recommendation(items, transactions, evaluation, similarity_measure):
    
    """
    
    This function is the main function that calls the other functions.
    It loops through all the items in the evaluation file and provides 
    the top-five recommendations. This function takes the following arguments:
    
    items: data-frame that contains all the books with their corresponding 
    engineered features.
    
    trasactions: data-frame that contains all the historical transactions.
    
    evaluation: data-frame that contains itemIDs for which we provide 
    recommendations.
    
    similarity_measure: possible values Euclidean, Cosine or Manhattan
    
    """
    
    ## Creating new colums for recommendations
    evaluation['rec_1'] = np.nan
    evaluation['rec_2'] = np.nan
    evaluation['rec_3'] = np.nan
    evaluation['rec_4'] = np.nan
    evaluation['rec_5'] = np.nan
    
    ## Extracting number of books in evaluation 
    n = evaluation.shape[0]
    
    for i in range(0, n):

        recommedations = recommendation(evaluation.loc[i, 'itemID'], transactions, items, similarity_measure)
        evaluation.loc[i, 'rec_1'] = recommedations[0]
        evaluation.loc[i, 'rec_2'] = recommedations[1]
        evaluation.loc[i, 'rec_3'] = recommedations[2]
        evaluation.loc[i, 'rec_4'] = recommedations[3]
        evaluation.loc[i, 'rec_5'] = recommedations[4]
    
    evaluation['rec_1'] = evaluation['rec_1'].astype(int)
    evaluation['rec_2'] = evaluation['rec_2'].astype(int)
    evaluation['rec_3'] = evaluation['rec_3'].astype(int)
    evaluation['rec_4'] = evaluation['rec_4'].astype(int)
    evaluation['rec_5'] = evaluation['rec_5'].astype(int)

    return evaluation
        

def recommendation(book, transactions, items, similarity_measure): 
    
    ## Declaring list to append potential recommendations
    book_to_recommend = []
    
    if (np.isin(book, [37378, 47675])):
        
        book_to_recommend = [55699, 78643, 23654, 58522, 74398]
        
        return book_to_recommend
    
    else:
        
        ## Extracting recommendations based on orders
        sessionID_order = transactions[(transactions['itemID'] == book) & (transactions['order'] > 0)]['sessionID'].values
        order_to_append = transactions[np.isin(transactions['sessionID'], sessionID_order)]['itemID'].values
        order_to_append = order_to_append[~np.isin(order_to_append, [book])]
        book_to_recommend.extend(order_to_append)    

        ## Extracting recommendations based on basket
        sessionID_basket = transactions[(transactions['itemID'] == book) & (transactions['basket'] > 0)]['sessionID'].values
        basket_to_append = transactions[np.isin(transactions['sessionID'], sessionID_basket)]['itemID'].values
        basket_to_append = basket_to_append[~np.isin(basket_to_append, [book])]
        book_to_recommend.extend(basket_to_append)

        ## Extracting recommendations based on click
        sessionID_click = transactions[(transactions['itemID'] == book) & (transactions['click'] > 0)]['sessionID'].values    
        click_to_append = transactions[np.isin(transactions['sessionID'], sessionID_click)]['itemID'].values
        click_to_append = click_to_append[~np.isin(click_to_append, [book])]
        book_to_recommend.extend(click_to_append)

        ## Extracting recommendation based on the above list 
        to_append_next_layer = np.array(recommendation_help(book_to_recommend, transactions))
        to_append_next_layer = to_append_next_layer[~np.isin(to_append_next_layer, [book])]
        book_to_recommend.extend(to_append_next_layer)

        ## Running k-NN (5 nearest neighbors)
        book_to_recommend = kNN.top_5_after_transaction(book, book_to_recommend, items, similarity_measure)
        
        return book_to_recommend

        
def recommendation_help(books, transactions):
    
    ## Defining list to store results 
    results =  []
    
    ## Determining the number of items
    n = len(books)
    
    for i in range(0, n):
        
        results.extend(recommendation_help_help(books[i], transactions))
        
    return results    


def recommendation_help_help(book, transactions): 
    
    ## Declaring list to append potential recommendations
    book_to_recommend = []
    
    ## Extracting recommendations based on orders
    sessionID_order = transactions[(transactions['itemID'] == book) & (transactions['order'] > 0)]['sessionID'].values
    order_to_append = transactions[np.isin(transactions['sessionID'], sessionID_order)]['itemID'].values
    order_to_append = order_to_append[~np.isin(order_to_append, [book])]
    book_to_recommend.extend(order_to_append)    
        
    ## Extracting recommendations based on basket
    sessionID_basket = transactions[(transactions['itemID'] == book) & (transactions['basket'] > 0)]['sessionID'].values
    basket_to_append = transactions[np.isin(transactions['sessionID'], sessionID_basket)]['itemID'].values
    basket_to_append = basket_to_append[~np.isin(basket_to_append, [book])]
    book_to_recommend.extend(basket_to_append)
    
    ## Extracting recommendations based on click
    sessionID_click = transactions[(transactions['itemID'] == book) & (transactions['click'] > 0)]['sessionID'].values    
    click_to_append = transactions[np.isin(transactions['sessionID'], sessionID_click)]['itemID'].values
    click_to_append = click_to_append[~np.isin(click_to_append, [book])]
    book_to_recommend.extend(click_to_append)   
    
    return book_to_recommend
        
        
        
    
