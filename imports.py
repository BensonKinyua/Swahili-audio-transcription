import streamlit as st
import joblib
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the pre-trained ResNet-18 model saved as a .pkl file
model = joblib.load('model_CNN.pkl')

# Define image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Create a Streamlit web app
st.title("Swahili Transcription with ResNet-18")

# Create file uploader
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    if st.button("Transcribe"):
        # Use the ResNet-18 model for transcription
        model.eval()
        with torch.no_grad():
            outputs = model(preprocessed_image)
            # Post-process the model's output (e.g., convert to text)
            transcription = post_process_transcription(outputs)

        st.write(f"Transcription: {transcription}")

# Define a function to post-process the model's output (replace with your specific logic)
def post_process_transcription(outputs):
    # Assuming 'outputs' is a tensor with shape (N, C) where N is the number of words and C is the number of classes.
    
    # Convert the tensor to a list of strings
    words = [str(word) for word in outputs]
    
    return words
