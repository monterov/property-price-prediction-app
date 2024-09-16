import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cargar el modelo entrenado desde el archivo .pkl
model = joblib.load('lightgbm_model.pkl')

# Instrucciones para los usuarios
st.sidebar.title("Instrucciones")
st.sidebar.write("""
1. Esta aplicación predice el precio de alquiler por noche de las propiedades ubicadas en **Londres**.
2. Introduce los detalles de la propiedad en los campos correspondientes.
3. Selecciona el barrio de la lista de barrios disponibles.
4. Haz clic en el botón "Predecir Precio" para obtener el precio estimado por noche.
5. Los baños completos se computan como números enteros, mientras que los aseos se suman con 0,5.
""")

# Título de la aplicación
st.title('Predicción del Precio de Alquiler en Londres')

# Input del usuario para las características de la propiedad
st.header('Introduce las características de tu propiedad')

# Coordenadas aproximadas (se pueden conectar con los barrios de Londres)
latitude = st.number_input('Latitud', value=51.5074)
longitude = st.number_input('Longitud', value=-0.1278)

# Selección de barrio
neighbourhood = st.selectbox('Selecciona el Barrio', options=['Kensington', 'Chelsea', 'Westminster', 'Otros'])

# Capacidad de alojamiento
accommodates = st.slider('Capacidad de Alojamiento (número de huéspedes)', min_value=1, max_value=10, value=2)

# Número de dormitorios
bedrooms = st.slider('Número de Dormitorios', min_value=1, max_value=10, value=1)

# Número de baños
bathrooms = st.slider('Número de Baños', min_value=1.0, max_value=10.0, step=0.5, value=1.0)

# Número de camas
beds = st.slider('Número de Camas', min_value=1, max_value=10, value=1)

# Mínimo y máximo de noches de estancia
min_nights = st.number_input('Mínimo de Noches', min_value=1, value=1)
max_nights = st.number_input('Máximo de Noches', min_value=1, value=30)

# Selección del tipo de habitación
room_type = st.selectbox('Tipo de Habitación', options=['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'])

# Selección de amenidades (checkboxes)
amenities = st.multiselect('Selecciona las amenidades disponibles', 
    ['Aire Acondicionado', 'Anfitrión amigable', 'Café', 'Caja fuerte', 
    'Casa de planta baja', 'Cocina', 'Conexión a internet', 'Cuna', 
    'Extintores', 'Horno', 'Muebles de jardín', 'Nevera', 'Otros', 
    'Patio', 'Pequeños electrodomésticos', 'Plancha', 'Secadora', 'Wifi'],
    default=['Otros'])

# Botón para predecir el precio
if st.button('Predecir Precio'):
    # Crear el dataframe con los inputs del usuario
    user_input = pd.DataFrame({
        'latitude_listings_clean': [latitude],
        'longitude_listings_clean': [longitude],
        'minimum_nights_avg_ntm_normalized': [min_nights],
        'maximum_nights_avg_ntm_normalized': [max_nights],
        'accommodates_normalized': [accommodates],
        'bathrooms_normalized': [bathrooms],
        'bedrooms_normalized': [bedrooms],
        'beds_normalized': [beds],
        'year': [2024],  # Asumimos que es el año actual
        'month': [9],  # Asumimos el mes actual
        'day': [15],  # Asumimos el día actual
        'Aire Acondicionado': [1 if 'Aire Acondicionado' in amenities else 0],
        'Anfitrión amigable': [1 if 'Anfitrión amigable' in amenities else 0],
        'Café': [1 if 'Café' in amenities else 0],
        'Caja fuerte': [1 if 'Caja fuerte' in amenities else 0],
        'Casa de planta baja': [1 if 'Casa de planta baja' in amenities else 0],
        'Cocina': [1 if 'Cocina' in amenities else 0],
        'Conexión a internet': [1 if 'Conexión a internet' in amenities else 0],
        'Cuna': [1 if 'Cuna' in amenities else 0],
        'Extintores': [1 if 'Extintores' in amenities else 0],
        'Horno': [1 if 'Horno' in amenities else 0],
        'Muebles de jardín': [1 if 'Muebles de jardín' in amenities else 0],
        'Nevera': [1 if 'Nevera' in amenities else 0],
        'Otros': [1 if 'Otros' in amenities else 0],
        'Patio': [1 if 'Patio' in amenities else 0],
        'Pequeños electrodomésticos': [1 if 'Pequeños electrodomésticos' in amenities else 0],
        'Plancha': [1 if 'Plancha' in amenities else 0],
        'Secadora': [1 if 'Secadora' in amenities else 0],
        'Wifi': [1 if 'Wifi' in amenities else 0]
    })

# Cargar el modelo
def load_model():
    model = joblib.load('lightgbm_model.pkl')
    return model

    # Predecir el precio con el modelo
    predicted_price = model.predict(user_input)[0]

    # Mostrar el resultado
    st.success(f'El precio estimado por noche es: ${predicted_price:.2f}')

    # Guardar la predicción en un archivo CSV
    csv_data = user_input
    csv_data['Predicted_Price'] = predicted_price
    csv_data.to_csv('prediccion_precio.csv', index=False)
    
    # Opción para que el usuario descargue el archivo CSV
    st.download_button(label="Descargar Predicción", data=csv_data.to_csv(index=False), file_name="prediccion_precio.csv", mime='text/csv')

