import joblib
import streamlit as st

# Diccionario de barrios con sus coordenadas (latitud, longitud)
barrios_coordenadas = {
    "Chelsea": (51.4875, -0.1680),
    "Kensington": (51.5074, -0.1877),
    "Mayfair": (51.5121, -0.1477),
    "Camden": (51.5413, -0.1419),
    "Soho": (51.5136, -0.1365),
    "Covent Garden": (51.5129, -0.1243),
    "Notting Hill": (51.5097, -0.2055),
    "Paddington": (51.5155, -0.1756),
    "Westminster": (51.4995, -0.1248),
    "Battersea": (51.4695, -0.1674),
    "Islington": (51.5380, -0.1032),
    "Shoreditch": (51.5265, -0.0782),
    "Hackney": (51.5450, -0.0552),
    "Whitechapel": (51.5193, -0.0599),
    "Southwark": (51.5035, -0.0804),
    "Greenwich": (51.4934, 0.0098),
    "Wimbledon": (51.4237, -0.2084),
    "Hampstead": (51.5566, -0.1784),
    "Clapham": (51.4613, -0.1389),
    "Fulham": (51.4821, -0.1997),
    "Hammersmith": (51.4927, -0.2237),
    "Richmond": (51.4613, -0.3037),
    "Chiswick": (51.4942, -0.2552),
    "Barnes": (51.4762, -0.2382),
    "Putney": (51.4591, -0.2179),
    "Victoria": (51.4964, -0.1430),
    "Kings Cross": (51.5300, -0.1232),
    "Bloomsbury": (51.5238, -0.1286),
    "Canary Wharf": (51.5054, -0.0235),
    "Woolwich": (51.4890, 0.0699),
    "Ealing": (51.5123, -0.3087),
    "Shepherd's Bush": (51.5045, -0.2171),
    "Acton": (51.5094, -0.2686),
    "Walthamstow": (51.5853, -0.0198),
    "Leytonstone": (51.5716, 0.0083),
    "Tottenham": (51.5881, -0.0696),
    "Brixton": (51.4651, -0.1148),
    "Peckham": (51.4734, -0.0662),
    "Lewisham": (51.4613, -0.0104),
    "Deptford": (51.4817, -0.0266),
    "Hoxton": (51.5312, -0.0839),
    "Finsbury Park": (51.5645, -0.1041),
    "St John's Wood": (51.5346, -0.1752),
    "Maida Vale": (51.5299, -0.1881),
    "Belsize Park": (51.5503, -0.1647)
}

# Cargar el modelo entrenado
model_path = "ruta_al_modelo_guardado.pkl"  # Cambia esta ruta por el modelo correcto
model = joblib.load(model_path)

# Mensaje para verificar que el modelo ha sido cargado correctamente
st.write("Modelo cargado con éxito.")

# Instrucciones para el usuario
st.sidebar.title("Instrucciones")
st.sidebar.write("""
1. Esta aplicación predice el precio de alquiler por noche de las propiedades ubicadas en Londres.
2. Introduce los detalles de la propiedad en los campos correspondientes.
3. Selecciona el barrio de la lista de barrios disponibles.
4. Haz clic en el botón "Predecir Precio" para obtener el precio estimado por noche.
5. Los baños completos se computan como números enteros, mientras que los aseos se suman con 0.5.
""")

# Lista desplegable para seleccionar el barrio
barrio_seleccionado = st.selectbox(
    "Selecciona el barrio:",
    list(barrios_coordenadas.keys())
)

# Obtener las coordenadas del barrio seleccionado
latitude_listings, longitude_listings = barrios_coordenadas[barrio_seleccionado]

# Otros campos que el usuario puede llenar
min_nights = st.number_input("Número mínimo de noches", value=1)
max_nights = st.number_input("Número máximo de noches", value=30)
room_type_shared = st.checkbox("Habitación Compartida", value=False)
room_type_private = st.checkbox("Habitación Privada", value=False)
accommodates = st.number_input("Número de huéspedes", value=2)
bathrooms = st.number_input("Número de baños", value=1)
bedrooms = st.number_input("Número de dormitorios", value=1)
beds = st.number_input("Número de camas", value=1)
# Aquí puedes agregar más campos como amenities y otros detalles

# Función para realizar la predicción
def hacer_prediccion(model, user_input):
    try:
        # Realizar la predicción
        predicted_price = model.predict([user_input])[0]
        return predicted_price
    except Exception as e:
        st.error(f"Error al predecir el precio: {str(e)}")
        return None

# Botón para predecir el precio
if st.button('Predecir Precio'):
    # Recoger los datos del formulario del usuario
    user_input = [
        latitude_listings, longitude_listings, min_nights, max_nights,
        room_type_shared, room_type_private, accommodates,
        bathrooms, bedrooms, beds,
        # Agrega más campos según sea necesario
    ]
    
    # Hacer la predicción
    predicted_price = hacer_prediccion(model, user_input)
    
    # Mostrar el precio predicho o un mensaje de error si falla
    if predicted_price is not None:
        st.success(f"El precio estimado por noche es: ${predicted_price:.2f}")
    else:
        st.error("No se pudo predecir el precio. Por favor, verifica la entrada.")
