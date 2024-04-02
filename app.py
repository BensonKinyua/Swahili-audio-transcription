# app.py
import streamlit as st
import joblib  # Import joblib to load the Pickle model
import asyncio
import threading
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

# Load the Pickle model
model = joblib.load('model_CNN.pkl')

import torchvision.transforms as transforms

def preprocess_image(image):
    # Define a series of transformations to be applied to the image
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the image to a square of size 256x256
        transforms.CenterCrop(224),  # Crop the center of the image to obtain a square of size 224x224
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image tensor
    ])
    
    # Apply the transformations to the image and unsqueeze it to add a batch dimension
    return transform(image).unsqueeze(0)

# Create input components (e.g., text input, sliders, etc.)
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


data_df = pd.DataFrame({'user_input': [image_file]})
if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    if st.button("Predict"):
        # Use the ResNet model for prediction
        model.eval()
        with torch.no_grad():
            outputs = model(preprocessed_image)
            _, predicted_class = torch.max(outputs, 1)
        
        st.write(f"Predicted class: {predicted_class.item()}")

