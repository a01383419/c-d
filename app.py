import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cats_vs_dogs_model.keras', compile=False)
    return model

model = load_model()
class_names = ['Cat', 'Dog']

st.title("Cats vs Dogs Classifier")
st.write("Upload an image to classify it as a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = image.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    label = class_names[int(prediction[0] > 0)]

    st.write(f"Prediction: {label}")
