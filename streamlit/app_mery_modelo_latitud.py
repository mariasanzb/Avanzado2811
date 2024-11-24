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
    # Crear DataFrame con nombres explícitos
    user_location = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude]})
    # Predecir el cluster
    cluster = kmeans.predict(user_location[['latitude', 'longitude']])[0]
    # Filtrar los mejores 5 restaurantes
    return df[df["cluster"] == cluster].iloc[0:5][['name', 'latitude', 'longitude', 'stars', 'categories', 'review_count', 'business_id']]

# Inicializar session_state para recomendaciones
if "recomendaciones" not in st.session_state:
    st.session_state.recomendaciones = None

# Entrada de usuario
st.sidebar.header("Ingresar Ubicación")
latitud = st.sidebar.number_input("Latitud", value=0.0, format="%.6f")
longitud = st.sidebar.number_input("Longitud", value=0.0, format="%.6f")

if st.sidebar.button("Buscar Restaurantes"):
    # Obtener recomendaciones
    st.session_state.recomendaciones = recommend_restaurants(top_restaurants, latitud, longitud, kmeans)

# Mostrar resultados si existen recomendaciones
if st.session_state.recomendaciones is not None:
    recomendaciones = st.session_state.recomendaciones

    if recomendaciones.empty:
        st.warning("No se encontraron restaurantes cercanos. Intenta con otra ubicación.")
    else:
        st.success(f"Estos son los {len(recomendaciones)} restaurantes que te recomendamos cerca de tu ubicacion")
        
        # Mostrar tabla de recomendaciones
        st.dataframe(recomendaciones[['name', 'latitude', 'longitude', 'stars', 'categories']])

        # Crear mapa solo una vez
        with st.spinner("Generando mapa..."):
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
