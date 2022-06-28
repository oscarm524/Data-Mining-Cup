import boto3
import pandas as pd
import numpy as np
import datetime
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from Features_Target_Funs import Surv_Features_1

s3 = boto3.resource('s3')
bucket_name = 'dmc-2022'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key = 'orders_items_parent.csv'

bucket_object = bucket.Object(file_key)
file_object = bucket_object.get()
file_content_stream = file_object.get('Body')

## Reading data file 
orders_items = pd.read_csv(file_content_stream)
orders_items['date'] = pd.to_datetime(orders_items['date'], format = '%Y-%m-%d')

## Defining the train and test
train = orders_items[orders_items['date'] < '2021-01-04'].reset_index(drop = True)
test = orders_items[orders_items['date'] > '2021-01-03'].reset_index(drop = True)

## Creating identifier for userID and itemID
train['user_item'] = train['userID'].apply(str) + '_' + train['itemID'].apply(str)
test['user_item'] = test['userID'].apply(str) + '_' + test['itemID'].apply(str)

## Features 1
user_item_surv = Surv_Features_1(train, test)
user_item_surv.to_csv('survival_modeling_full_data.csv', index = False)