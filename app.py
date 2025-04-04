import streamlit as st
import joblib
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import whisper
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import which
import os

# Set ffmpeg and ffprobe paths for pydub
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# Check if ffmpeg and ffprobe are available
if not AudioSegment.converter or not AudioSegment.ffprobe:
    st.error("ffmpeg or ffprobe not found. Make sure it's installed and accessible.")
    st.stop()

# Install ffmpeg for Linux environments (useful in some deployments)
os.system("apt-get update && apt-get install -y ffmpeg")

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load Smishing model and vectorizer
try:
    smishing_model = joblib.load("svm_smishing_model.pkl")
    smishing_vectorizer = joblib.load("tfidf_smishing_vectorizer.pkl")
except FileNotFoundError:
    st.error("Smishing model or vectorizer not found. Please ensure files are in the directory.")
    st.stop()

# Load Vishing model and vectorizer
try:
    vishing_model = joblib.load("svm_vishing_model.pkl")
    vishing_vectorizer = joblib.load("tfidf_vishing_vectorizer.pkl")
except FileNotFoundError:
    st.error("Vishing model or vectorizer not found. Please ensure files are in the directory.")
    st.stop()

# Load Whisper model for audio transcription
try:
    whisper_model = whisper.load_model("base")
except Exception as e:
    st.error(f"Failed to load Whisper model: {e}")
    st.stop()

# Page setup
st.set_page_config(page_title="Smishing & Vishing Detector", page_icon="🔒")

# Alert sound function
ALERT_SOUND_URL = "https://github.com/ThanujaBalla/smishing-vishing-detector/blob/main/alert.wav"
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

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Smishing prediction
def predict_smishing(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = smishing_vectorizer.transform([cleaned_text])
    prediction = smishing_model.predict(vectorized_text.toarray())[0]
    return prediction  # 1 = Smishing, 0 = Not Smishing

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

# Vishing prediction
def predict_vishing(audio_file):
    # Save uploaded file
    file_extension = audio_file.name.split(".")[-1].lower()
    temp_input = f"input_audio.{file_extension}"
    with open(temp_input, "wb") as f:
        f.write(audio_file.getbuffer())

    if file_extension == "mp3":
        temp_wav = "converted_audio.wav"
        convert_mp3_to_wav(temp_input, temp_wav)
        os.remove(temp_input)
    elif file_extension == "wav":
        temp_wav = temp_input
    else:
        st.error("Unsupported audio format. Please upload an MP3 or WAV file.")
        return None, None

    # Transcribe
    result = whisper_model.transcribe(temp_wav)
    transcribed_text = result["text"]

    # Predict
    cleaned_text = preprocess_text(transcribed_text)
    vectorized_text = vishing_vectorizer.transform([cleaned_text])
    prediction = vishing_model.predict(vectorized_text.toarray())[0]

    os.remove(temp_wav)
    return prediction, transcribed_text

# Streamlit UI
st.title("🔒 AI-Driven Smishing & Vishing Detection")
st.subheader("Protecting Your Financial Security")
st.write("Detect spam in text messages (smishing) or audio calls (vishing) with real-time alerts.")

# Input selection
st.markdown("### Choose Detection Type")
option = st.selectbox("Select an option:", ("Text (Smishing)", "Audio (Vishing)"), label_visibility="collapsed")

# Smishing detection
if option == "Text (Smishing)":
    st.markdown("#### Enter Text Message")
    text_input = st.text_area("Paste your message here:", height=150, placeholder="e.g., 'You won $1000, click here!'")
    if st.button("Analyze Text", key="smishing_btn"):
        if text_input.strip():
            with st.spinner("Analyzing message..."):
                prediction = predict_smishing(text_input)
                if prediction == 1:
                    st.error("⚠️ **Smishing Detected!** This message may be a scam.")
                    try:
                        play_alert()
                    except:
                        st.warning("Audio alert failed.")
                else:
                    st.success("✅ **Safe Message.** No smishing detected.")
        else:
            st.warning("Please enter a message to analyze.")

# Vishing detection
elif option == "Audio (Vishing)":
    st.markdown("#### Upload Audio File")
    audio_file = st.file_uploader("Upload an MP3 or WAV file:", type=["mp3", "wav"])
    if audio_file:
        if st.button("Analyze Audio", key="vishing_btn"):
            with st.spinner("Transcribing and analyzing audio..."):
                prediction, transcribed_text = predict_vishing(audio_file)
                if transcribed_text:
                    st.markdown("**Transcribed Text:**")
                    st.write(transcribed_text)
                if prediction == 1:
                    st.error("⚠️ **Vishing Detected!** This audio may be a scam.")
                    try:
                        play_alert()
                    except:
                        st.warning("Audio alert failed.")
                elif prediction == 0:
                    st.success("✅ **Safe Audio.** No vishing detected.")
    else:
        st.warning("Please upload an audio file to analyze.")

# Footer
st.markdown("---")
st.write("Developed by Team 23 | April 2025")
