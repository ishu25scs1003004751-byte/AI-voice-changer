import os
import streamlit as st
from groq import Groq
from gtts import gTTS
import tempfile
import numpy as np
from pydub import AudioSegment

# Page Config
st.set_page_config(page_title="AI Voice Converter", page_icon="🎙️")

st.title("🎙️ Real AI Voice Converter")
st.write("Apni voice upload karein aur pitch-shifted voice me convert karein!")

# Pitch Shift Helper Function using PyDub
def apply_voice_effect(audio_path, voice_style):
    sound = AudioSegment.from_file(audio_path)
    
    if voice_style == "Female Voice":
        # Pitch High (Female tone)
        octaves = 0.35
        new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
        transformed = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
        transformed = transformed.set_frame_rate(sound.frame_rate)
        
    elif voice_style == "Child Voice":
        # Super High Pitch + Slightly Faster Speed
        octaves = 0.55
        new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
        transformed = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
        transformed = transformed.set_frame_rate(sound.frame_rate).speedup(playback_speed=1.15)

    elif voice_style == "Monster Voice":
        # Deep Pitch Shift
        octaves = -0.4
        new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
        transformed = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
        transformed = transformed.set_frame_rate(sound.frame_rate)

    elif voice_style == "Robot Voice":
        # High Pitch + Fast Repeats / Metallic
        octaves = 0.1
        new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
        transformed = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
        transformed = transformed.set_frame_rate(sound.frame_rate)

    else:
        transformed = sound

    output_path = "final_pitch_shifted.mp3"
    transformed.export(output_path, format="mp3")
    return output_path

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
            # 1. Speech-to-Text
            with st.spinner("Processing Audio (Whisper STT)..."):
                with open(input_audio_path, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        file=(input_audio_path, file.read()),
                        model="whisper-large-v3",
                        response_format="text"
                    )
                st.info(f"**Recognized Text:** {transcription}")

            # 2. Text Adaption
            with st.spinner("Adapting Style with Llama..."):
                system_prompt = f"Rewrite this sentence to fit a {voice_option} tone. Keep output brief."
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": transcription}
                    ],
                    model="openai/gpt-oss-120b",
                )
                styled_text = chat_completion.choices[0].message.content.strip()
                st.success(f"**Adapted Script:** {styled_text}")

            # 3. Base TTS Generation
            with st.spinner("Applying Voice & Pitch Transformation..."):
                temp_raw_audio = "temp_raw.mp3"
                tts = gTTS(text=styled_text, lang="ur")
                tts.save(temp_raw_audio)

                # 4. Pitch Processing
                final_audio = apply_voice_effect(temp_raw_audio, voice_option)

                st.subheader("🎧 Transformed Real Voice:")
                st.audio(final_audio, format="audio/mp3")

        except Exception as e:
            st.error(f"Error: {str(e)}")
