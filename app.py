import os
import streamlit as st
from groq import Groq
from gtts import gTTS
import tempfile
import numpy as np
from scipy.io import wavfile
import scipy.signal

# Page Config
st.set_page_config(page_title="AI Voice Converter", page_icon="🎙️")

st.title("🎙️ Real AI Voice Converter")
st.write("Apni voice upload karein aur transformed voice me convert karein!")

# Audio Pitch Transformation Logic using Scipy
def apply_pitch_and_speed(input_wav_path, voice_style):
    sample_rate, audio_data = wavfile.read(input_wav_path)
    
    # Dual channel to single channel conversion
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    if voice_style == "Female Voice":
        # Pitch Increase (~40% pitch up)
        pitch_factor = 1.35
        resampled_data = scipy.signal.resample(audio_data, int(len(audio_data) / pitch_factor))
        output_rate = sample_rate
        
    elif voice_style == "Child Voice":
        # High Pitch (~60% pitch up)
        pitch_factor = 1.55
        resampled_data = scipy.signal.resample(audio_data, int(len(audio_data) / pitch_factor))
        output_rate = sample_rate

    elif voice_style == "Monster Voice":
        # Deep Pitch (~35% pitch down)
        pitch_factor = 0.65
        resampled_data = scipy.signal.resample(audio_data, int(len(audio_data) / pitch_factor))
        output_rate = sample_rate

    elif voice_style == "Robot Voice":
        # Robot / Ring Modulated Filter Effect
        t = np.arange(len(audio_data)) / sample_rate
        carrier = np.sin(2 * np.pi * 120 * t)
        resampled_data = audio_data * carrier
        output_rate = sample_rate
    else:
        resampled_data = audio_data
        output_rate = sample_rate

    # Normalize audio data
    resampled_data = np.int16(resampled_data / np.max(np.abs(resampled_data)) * 32767)

    output_wav_path = "final_output.wav"
    wavfile.write(output_wav_path, output_rate, resampled_data)
    return output_wav_path

# API Key handling
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("Groq API Key set nahi hai! Kripya Streamlit Secrets me add karein.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

voice_option = st.selectbox(
    "Target Voice Style Select Karein:",
    ("Female Voice", "Child Voice", "Robot Voice", "Monster Voice")
)

uploaded_file = st.file_uploader("Audio File Upload Karein", type=["wav", "mp3", "mpeg", "m4a", "ogg", "aac"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_audio_path = tmp_file.name

    st.audio(input_audio_path, format="audio/wav")

    if st.button("🚀 Convert Voice"):
        try:
            # 1. Speech-to-Text via Whisper
            with st.spinner("Processing Audio (Whisper STT)..."):
                with open(input_audio_path, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        file=(input_audio_path, file.read()),
                        model="whisper-large-v3",
                        response_format="text"
                    )
                st.info(f"**Recognized Text:** {transcription}")

            # 2. Text Adaptation via Llama
            with st.spinner("Adapting Style with Llama..."):
                system_prompt = f"Rewrite this text to sound like a {voice_option}. Output ONLY the rewritten text without commentary."
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": transcription}
                    ],
                    model="openai/gpt-oss-120b",
                )
                styled_text = chat_completion.choices[0].message.content.strip()
                st.success(f"**Adapted Script:** {styled_text}")

            # 3. Base TTS Audio Generation
            with st.spinner("Generating Transformed Audio Output..."):
                temp_mp3_path = "temp_speech.mp3"
                temp_wav_path = "temp_speech.wav"

                # Generate TTS
                tts = gTTS(text=styled_text, lang="ur")
                tts.save(temp_mp3_path)

                # Convert TTS MP3 to WAV for Scipy processing
                os.system(f"ffmpeg -i {temp_mp3_path} -y {temp_wav_path}")

                # Check if ffmpeg is available; fallback to standard audio output
                if os.path.exists(temp_wav_path):
                    final_audio_path = apply_pitch_and_speed(temp_wav_path, voice_option)
                    st.subheader("🎧 Transformed Voice Output:")
                    st.audio(final_audio_path, format="audio/wav")
                else:
                    st.subheader("🎧 Transformed Voice Output:")
                    st.audio(temp_mp3_path, format="audio/mp3")

        except Exception as e:
            st.error(f"Error: {str(e)}")
