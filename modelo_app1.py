import joblib
import requests
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime 

# URL cruda de GitHub al archivo del modelo (pkl)
url = 'https://raw.githubusercontent.com/monterov/property-price-prediction-app/main/lightgbm_model.pkl'

model_path = "lightgbm_model.pkl"

# Descargar el archivo del modelo desde GitHub
try:
    st.write("Descargando el modelo...")
    response = requests.get(url)
    response.raise_for_status()  # Verifica si la descarga tuvo éxito

    # Guardar el archivo en la ruta local
    with open(model_path, 'wb') as f:
        f.write(response.content)
    st.write("Modelo descargado y guardado exitosamente.")

except requests.exceptions.RequestException as e:
    st.error(f"Error al descargar el modelo: {str(e)}")

# Cargar el modelo desde la ruta local
try:
    st.write("Cargando el modelo...")
    model = joblib.load(model_path)
    st.write("Modelo cargado exitosamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")

# Verificar si el modelo se cargó correctamente
if 'model' not in locals():
    st.error("El modelo no se pudo cargar.")
    st.stop()

# Diccionario con los barrios de Londres y sus coordenadas (latitud y longitud)
barrios = {
    'Camden': (51.541, -0.142),
    'Greenwich': (51.482, -0.009),
    'Hackney': (51.545, -0.055),
    'Hammersmith and Fulham': (51.492, -0.233),
    'Islington': (51.538, -0.101),
    'Kensington and Chelsea': (51.499, -0.193),
    'Lambeth': (51.457, -0.116),
    'Lewisham': (51.461, -0.011),
    'Southwark': (51.503, -0.080),
    'Tower Hamlets': (51.509, -0.014),
    'Wandsworth': (51.457, -0.192),
    'Westminster': (51.497, -0.137),
    'Barnet': (51.653, -0.199),
    'Brent': (51.558, -0.282),
    'Ealing': (51.517, -0.306),
    'Haringey': (51.590, -0.111),
    'Hounslow': (51.470, -0.358),
}

# Instrucciones para los usuarios
st.sidebar.title("Instrucciones")
st.sidebar.write("""
1. Esta aplicación predice el precio de alquiler por noche de las propiedades ubicadas en **Londres**.
2. Introduce los detalles de la propiedad en los campos correspondientes.
3. Selecciona el barrio de la lista de barrios disponibles.
4. Haz clic en el botón "Predecir Precio" para obtener el precio estimado por noche.
5. Los baños completos se computan como números enteros, mientras que los aseos se suman con 0,5.
""")

# Variables de entrada para el usuario
st.title("Predicción de Precios de Alquiler en Londres")

# Selección del barrio
selected_barrio = st.selectbox("Selecciona el barrio", list(barrios.keys()))
latitude, longitude = barrios[selected_barrio]

st.write(f"Coordenadas del barrio seleccionado: Latitud {latitude}, Longitud {longitude}")

minimum_nights = st.number_input("Número mínimo de noches", min_value=1, max_value=365, value=3)
maximum_nights = st.number_input("Número máximo de noches", min_value=1, max_value=365, value=30)
room_type = st.selectbox("Tipo de habitación", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
accommodates = st.number_input("Capacidad de alojamiento (número de personas)", min_value=1, max_value=16, value=2)
bathrooms = st.number_input("Número de baños", min_value=1.0, max_value=5.0, value=1.0, step=0.5)
bedrooms = st.number_input("Número de dormitorios", min_value=1, max_value=10, value=1)
beds = st.number_input("Número de camas", min_value=1, max_value=10, value=1)

# Selección de amenidades
st.write("Selecciona las amenidades de la propiedad")
amenities = {
    'Aire Acondicionado': st.checkbox('Aire Acondicionado'),
    'Anfitrión amigable': st.checkbox('Anfitrión amigable'),
    'Café': st.checkbox('Café'),
    'Caja fuerte': st.checkbox('Caja fuerte'),
    'Casa de planta baja': st.checkbox('Casa de planta baja'),
    'Cocina': st.checkbox('Cocina'),
    'Conexión a internet': st.checkbox('Conexión a internet'),
    'Cuna': st.checkbox('Cuna'),
    'Extintores': st.checkbox('Extintores'),
    'Horno': st.checkbox('Horno'),
    'Muebles de jardín': st.checkbox('Muebles de jardín'),
    'Nevera': st.checkbox('Nevera'),
    'Otros': st.checkbox('Otros', value=True),  # Marcado por defecto
    'Patio': st.checkbox('Patio'),
    'Pequeños electrodomésticos': st.checkbox('Pequeños electrodomésticos'),
    'Plancha': st.checkbox('Plancha'),
    'Secadora': st.checkbox('Secadora'),
    'Wifi': st.checkbox('Wifi'),
}

# Obtener la fecha actual
now = datetime.now()
year = now.year
month = now.month
day = now.day

# Convertir amenidades a formato binario para el modelo
amenities_data = np.array([int(value) for value in amenities.values()])

# Botón para predecir el precio
if st.button("Predecir Precio"):
    # Crear DataFrame de entrada para el modelo
    input_data = pd.DataFrame({
        'latitude_listings_clean': [latitude],
        'longitude_listings_clean': [longitude],
        'minimum_nights_avg_ntm_normalized': [minimum_nights / 365],
        'maximum_nights_avg_ntm_normalized': [maximum_nights / 365],
        'room_type_clean_Shared room': [1 if room_type == "Shared room" else 0],
        'room_type_clean_Private room': [1 if room_type == "Private room" else 0],
        'room_type_clean_Hotel room': [1 if room_type == "Hotel room" else 0],
        'room_type_clean_Entire home/apt': [1 if room_type == "Entire home/apt" else 0],
        'accommodates_normalized': [accommodates / 16],
        'bathrooms_normalized': [bathrooms / 5],
        'bedrooms_normalized': [bedrooms / 10],
        'beds_normalized': [beds / 10]
    })

    # Añadir las amenidades
    for i, (amenity, _) in enumerate(amenities.items()):
        input_data[amenity] = amenities_data[i]

    # Hacer la predicción
    try:
        prediction = model.predict(input_data)
        st.write(f"El precio estimado por noche es: **${prediction[0] * 100:.2f}**")
    except Exception as e:
        st.error(f"Error al realizar la predicción: {str(e)}")


