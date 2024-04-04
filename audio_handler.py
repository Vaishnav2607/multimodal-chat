import torch
from transformers import pipeline
# from datasets import load_dataset
import librosa
import io

def convert_bytes_to_audio(audio_bytes):
    audio_bytes = io.BytesIO(audio_bytes)
    audio, sample_rate = librosa.load(audio_bytes)
    print(sample_rate)
    return audio

def transcribe_audio(audio_bytes):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=device,
    )
    audio_array = convert_bytes_to_audio(audio_bytes)
    prediction = pipe(audio_array, batch_size=1)["text"]
    # " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
    return prediction