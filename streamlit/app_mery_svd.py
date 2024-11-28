iimport streamlit as st
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

# Título principal
st.title("🔎 Sistema de Recomendación de Restaurantes")
st.write(
    """
    Bienvenido al sistema de recomendación de restaurantes. Aquí puedes obtener recomendaciones personalizadas según tus preferencias de usuario.
    La predicción se realiza con un modelo de filtrado colaborativo que predice las calificaciones de los restaurantes en función de tus gustos.
    """
)

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

# Interfaz de usuario en Streamlit
st.header("🔑 Introduce tu ID de Usuario para obtener recomendaciones")

user_id = st.text_input("Usuario ID:", placeholder="Ingresa tu ID de usuario")

if user_id:
    st.write(f"Mostrando recomendaciones para el usuario con ID: **{user_id}**")
    
    # Generar las recomendaciones
    top_recommendations = generate_top_recommendations(user_id, model, restaurant, top_n=5)
    
    if top_recommendations:
        st.subheader("🌟 Top 5 Restaurantes Recomendados")

        for i, recommendation in enumerate(top_recommendations, start=1):
            st.markdown(f"**{i}. {recommendation['name']}**")
            st.write(f"   - **ID**: {recommendation['business_id']}")
            st.write(f"   - **Rating Predicho**: {recommendation['predicted_rating']:.2f}")
            st.write(f"   - **Estrellas**: {recommendation['stars']}")
            st.write(f"   - **Ubicación**: {recommendation['latitude']}, {recommendation['longitude']}")
            st.write("---")
    else:
        st.warning("No se encontraron recomendaciones para este usuario. Intenta con otro ID.")
else:
    st.info("Por favor, ingresa tu ID de usuario para obtener recomendaciones.")
    
# Agregar algunos detalles para una mejor UX
st.sidebar.header("🔧 Acerca del Sistema")
st.sidebar.write(
    """
    Este sistema utiliza un modelo de filtrado colaborativo basado en el algoritmo **SVD (Singular Value Decomposition)** para hacer recomendaciones de restaurantes.
    Las recomendaciones se personalizan según las calificaciones previas de los usuarios.
    """
)

st.sidebar.markdown("### 🚀 ¿Cómo funciona?")
st.sidebar.write(
    """
    1. Ingresa tu ID de usuario en el campo proporcionado.
    2. El modelo predice las calificaciones de restaurantes que aún no has visitado.
    3. El sistema muestra las mejores recomendaciones basadas en esas predicciones.
    """
)
