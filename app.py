import streamlit as st
import cv2
import numpy as np
#from PIL import Image
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model

import platform

st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes")
#st.write("Versión de Python:", platform.python_version())
#image = Image.open('OIG5.jpg')
#st.image(image, width=350)

import base64

with open("OIG5.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
st.html(f'<img src="data:image/jpeg;base64,{b64}" width="350">')


with st.sidebar:
    st.subheader("Usando un modelo entrenado en teachable Machine puedes Usarlo en esta app para identificar")
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0]>0.5:
      st.header('Juan, con Probabilidad: '+str( prediction[0][0] * 100) + '%' )
    if prediction[0][1]>0.5:
      st.header('Jhorman, con Probabilidad: '+str( prediction[0][1]) + '%' )
    if prediction[0][2]>0.5:
     st.header('Santiago, con Probabilidad: '+str( prediction[0][2]) + '%' )


