import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Check if the model file exists
model_path = 'artifacts/fer2013.h5'  # Update this path to the location of your model

if not os.path.exists(model_path):
    st.error("Model file not found! Please check the path.")
else:
    # Load the model
    model = load_model(model_path)

    # Check the model summary (optional)
    model.summary()

    # Streamlit interface
    st.title("Face Emotion Recognition")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((100, 100))  # Resize to 48x48
        image_array = np.array(image) / 255.0  # Normalize the image
        image_array = np.reshape(image_array, (1, 100, 100, 1))  # Reshape for the model

        if st.button("Predict"):
            predictions = model.predict(image_array)
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            predicted_emotion = emotion_labels[np.argmax(predictions)]
            st.write(f"Predicted Emotion: {predicted_emotion}")