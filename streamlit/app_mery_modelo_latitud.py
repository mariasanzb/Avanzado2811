# pip install streamlit folium streamlit-folium

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

# Configuración de la página
st.set_page_config(page_title="Recomendación de Restaurantes", layout="wide")

# Encabezado
st.title("🍽️ Recomendación de Restaurantes")
st.markdown("Proporciona una ubicación para obtener recomendaciones de restaurantes cercanos.")

# Cargar el modelo y los datos
with open('./Data/modelo_recomendacion.pkl', 'rb') as file:
    data = pickle.load(file)

kmeans = data['kmeans']
top_restaurants = data['df']

# Definir la función para recomendar restaurantes
def recommend_restaurants(df, latitude, longitude, kmeans):
    # Predecir el cluster para la latitud y longitud proporcionadas
    cluster = kmeans.predict(np.array([latitude, longitude]).reshape(1, -1))[0]
    # Filtrar los mejores 5 restaurantes del cluster
    return df[df["cluster"] == cluster].iloc[0:5][['name', 'latitude', 'longitude', 'stars', 'categories', 'review_count', 'business_id']]

# Entrada de usuario
st.sidebar.header("Ingresar Ubicación")
latitud = st.sidebar.number_input("Latitud", value=0.0, format="%.6f")
longitud = st.sidebar.number_input("Longitud", value=0.0, format="%.6f")

if st.sidebar.button("Buscar Restaurantes"):
    # Obtener recomendaciones
    recomendaciones = recommend_restaurants(top_restaurants, latitud, longitud, kmeans)
    
    if recomendaciones.empty:
        st.warning("No se encontraron restaurantes cercanos. Intenta con otra ubicación.")
    else:
        st.success(f"Se encontraron {len(recomendaciones)} restaurantes cerca de la ubicación ingresada.")
        
        # Mostrar tabla de recomendaciones
        st.dataframe(recomendaciones[['name', 'latitude', 'longitude', 'stars', 'categories']])

        # Crear mapa
        m = folium.Map(location=[latitud, longitud], zoom_start=13)
        folium.Marker([latitud, longitud], popup="Tu ubicación", icon=folium.Icon(color="blue")).add_to(m)
        
        for _, row in recomendaciones.iterrows():
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=f"{row['name']} - {row['stars']}⭐\nCategorías: {row['categories']}",
                icon=folium.Icon(color="green")
            ).add_to(m)

        # Mostrar mapa en Streamlit
        st_folium(m, width=800, height=500)
