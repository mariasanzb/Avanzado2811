# pip install streamlit folium streamlit-folium

import streamlit as st
import pickle
import pandas as pd
import folium
from streamlit_folium import st_folium

# Cargar el modelo y los datos serializados
with open('./Data/modelo_recomendacion.pkl', 'rb') as file:
    data = pickle.load(file)

kmeans = data['kmeans']
top_restaurants = data['df']  # DataFrame con datos de los restaurantes

# Configuración de la página
st.set_page_config(page_title="Recomendación de Restaurantes", layout="wide")

# Encabezado
st.title("🍽️ Recomendación de Restaurantes")
st.markdown("Proporciona una ubicación para obtener recomendaciones de restaurantes cercanos.")

# Entrada de usuario
latitud = st.number_input("Latitud", value=0.0, format="%.6f")
longitud = st.number_input("Longitud", value=0.0, format="%.6f")

if st.button("Buscar Restaurantes"):
    # Agrupar por latitud y longitud
    user_location = pd.DataFrame({'latitude': [latitud], 'longitude': [longitud]})
    cluster = kmeans.predict(user_location)[0]

    # Filtrar restaurantes en el mismo cluster
    recomendaciones = top_restaurants[top_restaurants['cluster'] == cluster]

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
