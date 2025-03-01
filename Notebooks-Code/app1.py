import torch 
import streamlit as st 
import sounddevice as sd
import torchaudio
import numpy as np
import scipy.io.wavfile as wav
import os
from typing import Tuple, Union
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset

# Load the trained model
MODEL_PATH = r'C:\Users\nehaj\OneDrive\Desktop\healthcare hackathon\Slurred-Speech-Recognition-DeepLearning\Best-Fit-TrainedModel\scripted_model.pt'

@st.cache_resource  # Cache model to avoid reloading every time
def load_model():
    model = torch.jit.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# Audio parameters
SAMPLE_RATE = 16000  # Hz
DURATION = 5  # seconds

st.title("🎙 Impaired Speech-to-Text Converter")
st.write("Click below to record your voice and transcribe it.")

# Function to record audio
def record_audio():
    st.info("Recording... Speak now!")
    audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
    sd.wait()
    st.success("Recording Complete!")

    filename = "input_audio.wav"
    
    # Convert float32 to int16
    wav.write(filename, SAMPLE_RATE, np.int16(audio_data * 32767))
    
    return filename

# Function to preprocess audio
def preprocess_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    waveform, sample_rate = torchaudio.load(file_path, format='wav')

    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = transform(waveform)

    # Ensure it is a 2D tensor (1, N)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    return waveform


def load_librispeech_item(fileid,path,ext_audio,ext_txt):
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    file_text = speaker_id + "-" + chapter_id + ext_txt
    file_text = os.path.join(path, speaker_id, chapter_id, file_text)

    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    waveform, sample_rate = torchaudio.load(file_audio)

    with open(file_text) as ft:
        for line in ft:
            fileid_text, utterance = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            raise FileNotFoundError("Translation not found for " + fileid_audio)


if st.button("🎤 Record & Transcribe"):
    audio_file = record_audio()

    st.info("Processing Audio...")
    input_data = preprocess_audio(audio_file)
    
    with torch.no_grad():
        output = model(input_data)

    # Convert output tensor to text
    if isinstance(output, torch.Tensor):
        try:
            transcription = "".join([chr(int(x)) for x in output[0] if x > 0])  # Convert ASCII values
        except:
            transcription = "Error in decoding output."
    else:
        transcription = str(output)
        
    st.subheader("📝 Transcribed Text:")
    st.write(transcription)


# import streamlit as st
# import sounddevice as sd
# import numpy as np
# import scipy.io.wavfile as wav
# import os

# # Audio parameters
# SAMPLE_RATE = 44100  # Hz
# DURATION = 5  # seconds
# AUDIO_FILENAME = "input_audio.wav"

# st.title("🎙 Impaired Speech-to-Text Converter")
# st.write("Click below to record your voice and transcribe it.")

# # Function to record audio
# def record_audio():
#     """Records audio from the microphone and saves it as a WAV file."""
#     st.info("Recording... Speak now!")
#     audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
#     sd.wait()
#     st.success("Recording Complete!")

#     # Convert float32 to int16 before saving as WAV
#     wav.write(AUDIO_FILENAME, SAMPLE_RATE, np.int16(audio_data * 32767))
    
#     return AUDIO_FILENAME

# if st.button("🎤 Record & Transcribe"):
#     audio_file = record_audio()

#     st.info("Processing Audio...")

#     # ✅ Hardcoded response instead of model output
#     transcription = "Hi"

#     st.subheader("📝 Transcribed Text:")
#     st.write(transcription)