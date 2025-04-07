import streamlit as st
import tensorflow as tf # type: ignore
import numpy as np
from PIL import Image
import cv2 # type: ignore
from streamlit_drawable_canvas import st_canvas # type: ignore

# Cargar modelo
model = tf.keras.models.load_model('modelo_postal_3.h5')

st.title("Reconocimiento de Dígitos - Código Postal")
st.markdown("Dibuja un número del 0 al 9 en la pizarra y presiona **Predecir**.")

# Crear pizarra
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data

    # Mostrar imagen
    st.image(img, caption="Imagen Dibujada", width=150)

    # Botón para predecir
    if st.button("Predecir"):
        # Procesar imagen: escala de grises, resize 28x28, normalizar
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Predecir
        prediction = model.predict(img)
        digit = np.argmax(prediction)

        st.success(f"🔢 El dígito predicho es: **{digit}**")
