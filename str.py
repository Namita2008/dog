
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐶",
    layout="centered"
)

st.title("🐶 Dog vs Cat Classification")
st.write("Upload an image to predict whether it is a Dog or Cat.")

# Load class names
with open("classes.json", "r") as f:
    class_names = json.load(f)

# Load trained model
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("cat.keras")

model = load_cnn_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload Dog or Cat Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", width=300)

    # Resize image
    img = img.resize((64, 64))

    # Convert into array
    x = np.array(img, dtype=np.float32)

    # Normalize image (only if required by training)
    x = x / 255.0

    # Add batch dimension
    x = np.expand_dims(x, axis=0)

    # Prediction button
    if st.button("Predict Image"):

        prediction = model.predict(x)

        predicted_index = np.argmax(prediction[0])

        predicted_class = class_names[predicted_index]

        confidence = np.max(prediction[0]) * 100

        st.success(f"Prediction: {predicted_class}")

        st.info(f"Confidence: {confidence:.2f}%")
