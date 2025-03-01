import torch 
import streamlit as st 
import sounddevice as sd
import torchaudio
import numpy as np
import scipy.io.wavfile as wav
import os
import torch.nn as nn
torchaudio.set_audio_backend("soundfile")
# Loading the torch model 
path = r'C:\Users\nehaj\OneDrive\Desktop\healthcare hackathon\Slurred-Speech-Recognition-DeepLearning\Best-Fit-TrainedModel\scripted_model.pt'
def load_model():
    model = torch.jit.load(path, map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

rate = 44100
duration = 5

st.title("🎙 Impaired Speech-to-Text Converter ")
st.write("Click below to record your voice and transcribe it.")

def record_audio():
    st.info("Recording... Speak now!")
    audio_data = sd.rec(int(rate * duration), samplerate=rate, channels=1, dtype=np.float32)
    sd.wait()
    st.success("Recording Complete!")
    
    filename = "input_audio.wav"
    wav.write(filename,rate, np.int16(audio_data * 32767))
    
    return filename

def preprocess_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    waveform, sample_rate = torchaudio.load(file_path, format='wav')
    
    if sample_rate != rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=rate)(waveform)

if st.button("🎤 Record & Transcribe"):
    audio_file = record_audio()

    st.info("Processing Audio...")
    input_data = preprocess_audio(audio_file)
    
    with torch.no_grad():
        predicted_text = model(input_data)

    if isinstance(predicted_text, torch.Tensor):
        transcription = "".join([chr(int(x)) for x in predicted_text[0]])
        
    else:
        transcription = str(predicted_text)
        
    st.subheader("📝 Transcribed Text:")
    st.write(transcription)