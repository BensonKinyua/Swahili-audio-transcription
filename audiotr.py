import streamlit as st
import joblib
import torch
import torchaudio
import torchaudio.transforms as transforms
import torch.nn.functional as F

# Load the pre-trained ResNet-18 model saved as a .pkl file
model = joblib.load('model_CNN.pkl')

# Define audio preprocessing function to create a spectrogram
def preprocess_audio(audio):
    # Load the audio file and convert it to a spectrogram
    waveform, sample_rate = torchaudio.load(audio)
    
    # Apply audio transformations to create a spectrogram
    transform = transforms.MelSpectrogram(sample_rate=sample_rate)
    spectrogram = transform(waveform)
    
    # Expand the spectrogram to add a batch dimension
    spectrogram = spectrogram.unsqueeze(0)
    
    return spectrogram

# Create a Streamlit web app
st.title("Swahili Transcription with ResNet-18")

# Create file uploader
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])  # Adjust the accepted audio formats as needed

if audio_file is not None:
    spectrogram = preprocess_audio(audio_file)

    if st.button("Transcribe"):
        # Use the ResNet-18 model for transcription
        model.eval()
        with torch.no_grad():
            outputs = model(spectrogram)
            # Post-process the model's output (e.g., convert to text)
            

        st.write(f"Transcription: {transcription}")

# Define a function to post-process the model's output (replace with your specific logic)
# Define a function to post-process the model's output
def post_process_transcription(outputs):
    # Assuming 'outputs' is a tensor with shape (N, C) where N is the number of words and C is the number of classes.
    
    # Find the index of the word with the highest probability
    predicted_word = torch.argmax(outputs, dim=1)
    
    # Convert the index to a text label (replace with your specific logic)
    labels = ['hapana',
                  'kumi',
                  'mbili',
                  'moja',
                  'nane',
                  'ndio',
                  'nne',
                  'saba',
                  'sita',
                  'tano',
                  'tatu',
                  'tisa']  # label names
    transcription = labels[predicted_word.item()]
    
    return transcription

transcription = post_process_transcription(outputs)
st.write(f"Transcription: {transcription}")