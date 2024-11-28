import streamlit as st
import pandas as pd
import pickle
import difflib
import gdown

# Cargar los datasets
url = 'https://drive.google.com/uc?export=download&id=1tY2U8IjcUo2NTbXMWvY1Bg9eG8qn9fcK'
gdown.download(url, 'reviews.parquet', quiet=False)

# Cargar los archivos Parquet con pandas
reviews = pd.read_parquet('reviews.parquet', engine='pyarrow')
restaurant = pd.read_parquet('mi_archivo1.parquet', engine='pyarrow')

# Funciones auxiliares
def get_restaurant_id(restaurant_title, metadata):
    existing_titles = list(metadata['name'].values)
    closest_titles = difflib.get_close_matches(restaurant_title, existing_titles)
    if not closest_titles:
        return None
    restaurant_id = metadata[metadata['name'] == closest_titles[0]]['business_id'].values[0]
    return restaurant_id

def get_restaurant_info(restaurant_id, metadata):
    restaurant_info = metadata[metadata['business_id'] == restaurant_id][
        ['business_id', 'latitude', 'longitude', 'name', 'stars']
    ]
    return restaurant_info.to_dict(orient='records')

def predict_review(user_id, restaurant_title, model, metadata):
    restaurant_id = get_restaurant_id(restaurant_title, metadata)
    if not restaurant_id:
        return None
    review_prediction = model.predict(uid=user_id, iid=restaurant_id)
    return review_prediction.est

def generate_top_recommendations(user_id, model, metadata, top_n=5, thresh=4):
    restaurant_titles = list(metadata['name'].values)
    recommendations = []

    for restaurant_title in restaurant_titles:
        rating = predict_review(user_id, restaurant_title, model, metadata)
        if rating and rating >= thresh:
            restaurant_id = get_restaurant_id(restaurant_title, metadata)
            restaurant_info = get_restaurant_info(restaurant_id, metadata)[0]
            restaurant_info['predicted_rating'] = rating
            recommendations.append(restaurant_info)
