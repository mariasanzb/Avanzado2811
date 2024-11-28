import streamlit as st
import pandas as pd
import pickle
from surprise import SVD, Dataset, Reader
import difflib
import gdown

# Cargar los datasets
url = 'https://drive.google.com/uc?export=download&id=1tY2U8IjcUo2NTbXMWvY1Bg9eG8qn9fcK'
gdown.download(url, 'reviews.parquet', quiet=False)

# Ahora puedes cargar el archivo como un DataFrame de pandas o usarlo como lo necesites
import pandas as pd
reviews = pd.read_parquet('reviews.parquet',engine='pyarrow')

restaurant = pd.read_parquet('mi_archivo1.parquet', engine='pyarrow')

# Crear el dataset de Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(reviews[['user_id', 'business_id', 'rating']], reader)

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
    
    recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
    return recommendations[:top_n]

# Cargar el modelo serializado
with open('./Data/svd_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Configurar la interfaz de Streamlit
st.title("Sistema de Recomendación de Restaurantes")
st.write("Obtén las mejores recomendaciones personalizadas para tus restaurantes favoritos.")

user_id = st.text_input("Introduce el ID del usuario:")
if user_id:
    st.write(f"Mostrando recomendaciones para el usuario: {user_id}")
    
    top_recommendations = generate_top_recommendations(user_id, model, restaurant, top_n=5)
    
    if top_recommendations:
        st.write("Top 5 restaurantes recomendados:")
        for i, recommendation in enumerate(top_recommendations, start=1):
            st.subheader(f"{i}. {recommendation['name']}")
            st.write(f"- ID: {recommendation['business_id']}")
            st.write(f"- Rating Predicho: {recommendation['predicted_rating']:.2f}")
            st.write(f"- Estrellas: {recommendation['stars']}")
            st.write(f"- Ubicación: {recommendation['latitude']}, {recommendation['longitude']}")
            st.write("---")
    else:
        st.write("No se encontraron recomendaciones para este usuario.")

