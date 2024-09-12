import os
import gdown
import lightgbm as lgb
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar las columnas del entrenamiento
columnas_entrenamiento = pd.read_csv('columnas_entrenamiento.csv').values.flatten()

# URL de tu archivo en Google Drive
file_url = "https://drive.google.com/uc?id=1BQAQosuYB6rR0jMIxowC61UwUm0DuZ3s"
output_file = 'modelo_lightgbm.txt'

# Descargar el archivo desde Google Drive
gdown.download(file_url, output_file, quiet=False)

# Definir la variable `local_filename`
local_filename = 'modelo_lightgbm.txt'  # Aquí defines el nombre del archivo descargado

# Cargar el modelo LightGBM descargado
model = lgb.Booster(model_file=local_filename)

# Instrucciones para los usuarios
st.sidebar.title("Instrucciones")
st.sidebar.write("""
1. Esta aplicación predice el precio de alquiler por noche de las propiedades ubicadas en **Londres**.
2. Introduce los detalles de la propiedad en los campos correspondientes.
3. Selecciona el barrio de la lista de barrios disponibles.
4. Haz clic en el botón "Predecir Precio" para obtener el precio estimado por noche.
5. Si estás considerado en la plataforma como "Superhost", pon "sí" en la casilla correspondiente.
6. Los baños completos se computan como números enteros, mientras que los aseos se suman con 0,5.
""")

# Crear la interfaz en Streamlit
st.title("Predicción de Precio de Propiedades en Londres")

# Crear los inputs en la interfaz
accommodates = st.number_input("Número de personas", min_value=1, max_value=16, value=2)
bathrooms = st.number_input("Número de baños", min_value=1.0, max_value=10.0, step=0.5, value=1.0)
bedrooms = st.number_input("Número de dormitorios", min_value=1, max_value=10, value=1)
beds = st.number_input("Número de camas", min_value=1, max_value=10, value=1)
superhost = st.selectbox("¿Es superhost?", options=["Sí", "No"])

# Listado completo de barrios de Londres
neighbourhood = st.selectbox("Barrio", options=[
    "Hackney", "Camden", "Islington", "Westminster", "Kensington and Chelsea", 
    "Lambeth", "Southwark", "Tower Hamlets", "Wandsworth", "Hammersmith and Fulham",
    "Greenwich", "Lewisham", "Bromley", "Croydon", "Sutton", "Merton", 
    "Richmond upon Thames", "Kingston upon Thames", "Haringey", "Barnet", 
    "Enfield", "Harrow", "Brent", "Ealing", "Hounslow", "Redbridge", 
    "Waltham Forest", "Havering", "Bexley", "Barking and Dagenham", "Newham"])

room_type = st.selectbox("Tipo de habitación", options=["Entire home/apt", "Private room", "Shared room"])
min_nights = st.number_input("Estancia mínima", min_value=1, max_value=365, value=1)
max_nights = st.number_input("Estancia máxima", min_value=1, max_value=365, value=365)
availability_30 = st.number_input("Disponibilidad en 30 días", min_value=0, max_value=30, value=10)
reviews = st.number_input("Número de reseñas", min_value=0, max_value=1000, value=10)

# Almacenar los datos en un DataFrame
datos_propiedad = pd.DataFrame({
    'accommodates_clean': [accommodates],
    'bathrooms_clean': [bathrooms],
    'bedrooms_clean': [bedrooms],
    'beds_clean': [beds],
    'host_is_superhost_clean': [superhost],
    'neighbourhood_cleansed': [neighbourhood],
    'room_type_clean': [room_type],
    'minimum_nights_avg_ntm_clean': [min_nights],
    'maximum_nights_avg_ntm_clean': [max_nights],
    'availability_30_clean': [availability_30],
    'number_of_reviews_clean': [reviews]
})

# Preprocesamiento para que coincidan las columnas
def preprocesar_datos(datos_propiedad, columnas_entrenamiento):
    datos_propiedad_processed = pd.get_dummies(datos_propiedad, 
                                               columns=['host_is_superhost_clean', 'neighbourhood_cleansed', 'room_type_clean'], 
                                               drop_first=True)

    datos_propiedad_processed['accommodates_bathrooms'] = datos_propiedad_processed['accommodates_clean'] * datos_propiedad_processed['bathrooms_clean']
    datos_propiedad_processed['bedrooms_bathrooms'] = datos_propiedad_processed['bedrooms_clean'] * datos_propiedad_processed['bathrooms_clean']
    
    # Reindexar para asegurar que las columnas coincidan con las del entrenamiento
    datos_propiedad_processed = datos_propiedad_processed.reindex(columns=columnas_entrenamiento, fill_value=0)

    # Verificar los datos que se están pasando al modelo
    st.write(datos_propiedad_processed)  # Esto imprimirá las características preprocesadas

    # Estandarización (usando un scaler si es necesario)
    scaler = StandardScaler()
    datos_propiedad_scaled = scaler.fit_transform(datos_propiedad_processed)
    
    return datos_propiedad_scaled

# Botón para predecir
if st.button("Predecir Precio"):
    # Preprocesar los datos para que coincidan con las columnas del entrenamiento
    datos_propiedad_scaled = preprocesar_datos(datos_propiedad, columnas_entrenamiento)
    
    # Predecir el precio
    precio_predicho_log = model.predict(datos_propiedad_scaled)
    precio_predicho = np.expm1(precio_predicho_log)  # Revertir la transformación logarítmica
    st.write(f"El precio predicho de la propiedad es: {precio_predicho[0]:.2f} € por noche")








