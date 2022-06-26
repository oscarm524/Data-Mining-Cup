# Data Mining Cup 2021

In this repository, we store all the code related to this competition. For more information about the 2021 data mining competition [click here](https://www.data-mining-cup.com/dmc-2021/). 

## Scenario
Before the pandemic, Johannes Gutenberg managed a flourishing little book shop in the historic city center of Mainz. He took great joy in building personal relationships with each of his clients, recommending books catered to their personal taste and assisting in widening their literary palette. In his city and beyond he developed a formidable reputation with a respectable base of loyal customers, who considered him more of a connoisseur than a traditional salesman.

Unfortunately, this loyal base of customers is not enough to make his business profitable. And so, like many traditional retailers, Johannes also relies on walk-in customers. At the beginning of the pandemic, this source of revenue vanished. To keep his employees and cover ongoing costs, Johannes had to find an alternative form of revenue. With great initial reservation, he decided to expand his business by launching an online shop, which he believed would save his beloved business from imminent bankruptcy. At first, Johannes and his employees tried their best to provide suitable recommendations for every product manually. But as the number of products increased and associates worked to keep at least some personal contact to clients via phone and email, this manual process was just not feasible.

Today, Johannes is looking for a reliable recommendation system to provide a targeted recommendation to every product page. This solution should meet his high personalization standards and only require a small amount of manual support to implement.

## The Task
The goal for each participating team is to create a recommendation model based on historical transactions and item features. For any given product, the model should return its five best recommendations. In order to create a recommender model, the participants are provided with historical transaction and descriptive item data in the form of structured text files (.csv).

## Feature Engineering
To begin, we used the [textblob language detection package](https://textblob.readthedocs.io/en/dev/) to determine the language of each book. This approach involves taking the book titles and using the detect_language function to output a two-letter language code (ISO 639-1 codes). All books with numerical titles or titles with less than four characters required a manual language search. 

Following the language detection process, we created several categorical variables to further distinguish the books from one another. With the table of [Thema Subject Categories](https://ns.editeur.org/thema/en) (provided in the problem statement), we created several variables that relate to the general topic (first character of main topic variable), the second general topic (first two characters of the main topic variable), and the third general topic (first three characters of the main topic variable, where applicable). Next, we created a dummy variable for the seven most common detected languages. And finally, we created a dummy variable for each of the 50 most popular authors in the data set (the most frequently occuring authors). The purpose of all create dummy variables is to be able to distinguish books from one another for our "k-Nearest Neighbor" approach using the calculated Euclidean distance, cosine similarity, or Manhanttandistance as measure of similarity between observations.

## First Approach
After the several features are engineered, we are ready to start implementing our first recommendation engine. The recommendation system relies on similarity, which can be measured with the Euclidean distance, cosine similarity or Manhattan distance, and transaction activity. 

```
PROGRAM Book_Recommendation:
  
  FOR each book in evaluation.csv
  
    IF book is not in transactions.csv
      THEN provide the five-nearest neighbors as the recommendations;
    
    ELSE 
      THEN extract book that were click, basket, or order in the same sessions;
      
        IF the number of extracted books from the previous step is greater than or equal to 5
          THEN provide the five-nearest neighbors as the recommendations;
        
        ELSE
          THEN provide the five-nearest neighbors as the recommendations;
          
        ENDIF;
    
    ENDIF;
  
  ENDFOR;

END.
```


## Second Approach
After the first approach, we implement our second recommendation engine, wich is an improvement of the first recommendation engine. The recommendation system relies on similarity, which can measured with the Euclidean distance, cosine similarity or Manhattan distance, and transaction activity. On top of that, we complement similarity with popularity score, on books with transaction activity, which was engineered by the means of outlier analysis techniques. Notice that we use the algorithm presented on page 78 of [Outlier Analysis](https://www.amazon.com/Outlier-Analysis-Charu-C-Aggarwal/dp/3319475770/ref=sr_1_1?dchild=1&keywords=outlier+analysis&qid=1624574912&s=books&sr=1-1)

```
PROGRAM Book_Recommendation_Rating:
  
  FOR each book in transaction.csv
    Compute their outlier socre using the algorithm presented on page 78 of Outlier Analysis. After that, we standardize these scores to 1-5 scale;
  
  ENDFOR;
    
  
  FOR each book in evaluation.csv
  
    IF book is not in transactions.csv
      THEN provide the five-nearest neighbors as the recommendations;
    
    ELSE 
      THEN extract book that were click, basket, or order in the same sessions (with their corresponding popularity scores);
      
        IF the number of extracted books from the previous step is greater than or equal to 20
          THEN provide the five-nearest neighbors with the highest popularity socres as the recommendations;
        
        ELSE
          THEN provide the five-nearest neighbors with the highest popularity scores as the recommendations;
          
        ENDIF;
    
    ENDIF;
  
  ENDFOR;

END.
```
