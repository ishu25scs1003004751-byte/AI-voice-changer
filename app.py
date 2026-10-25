import os
import streamlit as st
from groq import Groq
from gtts import gTTS
import tempfile

# Page Config
st.set_page_config(page_title="AI Voice Converter", page_icon="🎙️")

st.title("🎙️ AI Voice Converter")
st.write("Apni voice convert karne ke liye audio upload karein!")

# API Key handling
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("Groq API Key set nahi hai! Kripya Streamlit Secrets me add karein.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Voice Options
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
            # 1. Whisper Speech to Text
            with st.spinner("Processing Audio (Whisper STT)..."):
                with open(input_audio_path, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        file=(input_audio_path, file.read()),
                        model="whisper-large-v3",
                        response_format="text"
                    )
                st.info(f"**Recognized Text:** {transcription}")

            # 2. Llama Text Transformation
            with st.spinner("Modifying Tone with Llama..."):
                system_prompt = (
                    f"Rewrite the input text to sound like a {voice_option}. "
                    "Output ONLY the converted sentence without quotes or commentary."
                )
                
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": transcription}
                    ],
                    model="openai/gpt-oss-120b",
                )
                styled_text = chat_completion.choices[0].message.content.strip()
                st.success(f"**Adapted Script:** {styled_text}")

            # 3. Audio Synthesis (gTTS Fix for Multi-Language)
            with st.spinner("Generating Audio Output..."):
                output_audio_path = "output_transformed.mp3"
                
                # Auto-detect language or fallback to 'ur'/'hi'/'en'
                try:
                    tts = gTTS(text=styled_text, lang="ur", slow=(voice_option in ["Robot Voice", "Monster Voice"]))
                    tts.save(output_audio_path)
                except Exception:
                    tts = gTTS(text=styled_text, lang="hi", slow=(voice_option in ["Robot Voice", "Monster Voice"]))
                    tts.save(output_audio_path)

                st.subheader("🎧 Converted Audio:")
                st.audio(output_audio_path, format="audio/mp3")

        except Exception as e:
            st.error(f"Error: {str(e)}")
