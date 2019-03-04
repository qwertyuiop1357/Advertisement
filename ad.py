import time
import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


DATA_PATH = './data.txt'

POP_FEATURES = ['instance_id', 'item_property_list', 'user_id',
                'context_timestamp', 'context_id',
                'predict_category_property', 'date']

CATEGORICAL_FEATURES = ['item_id', 'item_brand_id', 'item_city_id', 'user_gender_id', 'user_occupation_id', 'shop_id']

CONTINUOUS_FEATURES = ['item_price_level', 'item_sales_level', 'item_collected_level',
                       'item_pv_level', 'user_age_level', 'user_star_level',
                       'shop_review_num_level', 'shop_review_positive_rate',
                       'shop_star_level', 'shop_score_service', 'shop_score_delivery',
                       'shop_score_description']

VECTOR_FEATURES = ['item_category_list']

