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

def _pop_features(df):
    for feature in POP_FEATURES:
        df.pop(feature)

def _generate_categorical_features(df, CATEGORICAL_FEATURES):
    """
    categorical features
    """
    oh_encoder = OneHotEncoder(sparse=True, categories='auto')
    space = oh_encoder.fit_transform(df[CATEGORICAL_FEATURES[0]].values.reshape((-1, 1))).toarray()
    for feature in CATEGORICAL_FEATURES[1:]:
        val = oh_encoder.fit_transform(df[feature].values.reshape((-1, 1))).toarray()
        space = np.hstack((space, val))
    return space


def _generate_vector_features(df, VECTOR_FEATURES):
    """
    vector features
    """
    cv = CountVectorizer()
    for feature in VECTOR_FEATURES:
        val = cv.fit_transform(df[feature]).toarray()
    return val


if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH, sep=' ', nrows=100000)
    df['date'] = df['context_timestamp'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))
    df = _generate_historical_convrate(df)
    df = _generate_instant_feature(df)
    y = df.pop('is_trade')
    _pop_features(df)
    df = _generate_categorical_features(df, CATEGORICAL_FEATURES)
    X = _generate_vector_features(df, VECTOR_FEATURES)
   
