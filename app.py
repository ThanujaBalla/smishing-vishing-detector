import streamlit as st
import joblib
import numpy as np
import nltk
import re
import shutil
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import whisper
from pydub import AudioSegment
import os

# Ensure ffmpeg is available
if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    st.error("FFmpeg or ffprobe not found. Please ensure they are installed and accessible.")
    st.stop()

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load models
try:
    smishing_model = joblib.load("svm_smishing_model.pkl")
    smishing_vectorizer = joblib.load("tfidf_smishing_vectorizer.pkl")
except FileNotFoundError:
    st.error("Smishing model/vectorizer not found.")
    st.stop()

try:
    vishing_model = joblib.load("svm_vishing_model.pkl")
    vishing_vectorizer = joblib.load("tfidf_vishing_vectorizer.pkl")
except FileNotFoundError:
    st.error("Vishing model/vectorizer not found.")
    st.stop()

# Load Whisper model
try:
    whisper_model = whisper.load_model("base")
except Exception as e:
    st.error(f"Failed to load Whisper model: {e}")
    st.stop()

# Alert playback JS (browser-based)
ALERT_SOUND_URL = "https://github.com/ThanujaBalla/smishing-vishing-detector/raw/main/alert.wav"
def play_alert():
    audio_script = f"""
        <script>
        function playSound() {{
            var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            var source = audioCtx.createBufferSource();
            fetch("{ALERT_SOUND_URL}")
            .then(response => response.arrayBuffer())
            .then(arrayBuffer => audioCtx.decodeAudioData(arrayBuffer))
            .then(audioBuffer => {{
                source.buffer = audioBuffer;
                source.connect(audioCtx.destination);
                source.start();
            }})
            .catch(error => console.error("Audio playback failed:", error));
        }}
        playSound();
        </script>
    """
    st.markdown(audio_script, unsafe_allow_html=True)

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Smishing detection
def predict_smishing(text):
    cleaned = preprocess_text(text)
    vec = smishing_vectorizer.transform([cleaned])
    pred = smishing_model.predict(vec.toarray())[0]
    return pred

# MP3 → WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

# Vishing detection
def predict_vishing(audio_file):
    ext = audio_file.name.split(".")[-1].lower()
    temp_input = f"input_audio.{ext}"
    with open(temp_input, "wb") as f:
        f.write(audio_file.getbuffer())

    temp_wav = "converted_audio.wav"
    if ext == "mp3":
        convert_mp3_to_wav(temp_input, temp_wav)
        os.remove(temp_input)
    elif ext == "wav":
        temp_wav = temp_input

    result = whisper_model.transcribe(temp_wav)
    transcribed = result["text"]

    cleaned = preprocess_text(transcribed)
    vec = vishing_vectorizer.transform([cleaned])
    pred = vishing_model.predict(vec.toarray())[0]

    os.remove(temp_wav)
    return pred, transcribed

# UI
st.set_page_config(page_title="Smishing & Vishing Detector", page_icon="🔒")
st.title("🔒 AI-Driven Smishing & Vishing Detection")
st.subheader("Protecting Your Financial Security")

option = st.selectbox("Choose Detection Type", ("Text (Smishing)", "Audio (Vishing)"))

if option == "Text (Smishing)":
    st.markdown("#### Enter Text Message")
    msg = st.text_area("Message:", height=150)
    if st.button("Analyze Text"):
        if msg.strip():
            with st.spinner("Analyzing..."):
                pred = predict_smishing(msg)
                if pred == 1:
                    st.error("⚠️ Smishing Detected!")
                    play_alert()
                else:
                    st.success("✅ Safe Message.")
        else:
            st.warning("Please enter a message.")

elif option == "Audio (Vishing)":
    st.markdown("#### Upload Audio File")
    audio = st.file_uploader("Upload MP3 or WAV", type=["mp3", "wav"])
    if audio:
        if st.button("Analyze Audio"):
            with st.spinner("Transcribing and analyzing..."):
                pred, transcript = predict_vishing(audio)
                st.markdown("**Transcribed Text:**")
                st.write(transcript)
                if pred == 1:
                    st.error("⚠️ Vishing Detected!")
                    play_alert()
                else:
                    st.success("✅ Safe Audio.")
    else:
        st.warning("Please upload an audio file.")

st.markdown("---")
st.caption("Developed by Team 23 | April 2025")
