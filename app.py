import os
import streamlit as st
from groq import Groq
import tempfile
import asyncio
import edge_tts

# Streamlit Page Config
st.set_page_config(page_title="AI Voice Converter", page_icon="🎙️")

st.title("🎙️ Real Human AI Voice Converter")
st.write("Apni voice upload karein aur ultra-realistic human voices me convert karein!")

# API Key Check
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("Groq API Key set nahi hai! Kripya Streamlit Secrets me add karein.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Edge-TTS Natural Human Voice Mapping (High-Reliability Voices)
VOICE_MAPPING = {
    "Female Voice": "hi-IN-SwaraNeural",         # Natural Female Voice (Hindi/Urdu Support)
    "Child Voice": "hi-IN-MadhurNeural",         # Natural Young Voice
    "Monster Voice": "en-US-ChristopherNeural",   # Deep Heavy Male Voice
    "Robot Voice": "en-US-GuyNeural"             # Standard Male Voice
}

# Async Edge-TTS Function
async def generate_natural_voice(text, voice_name, output_file):
    communicate = edge_tts.Communicate(text, voice_name)
    await communicate.save(output_file)

# UI Elements
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
            # 1. Speech to Text via Whisper
            with st.spinner("Processing Audio (Whisper STT)..."):
                with open(input_audio_path, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        file=(input_audio_path, file.read()),
                        model="whisper-large-v3",
                        response_format="text"
                    )
                st.info(f"**Recognized Text:** {transcription}")

            # 2. Text Modification via Llama
            with st.spinner("Adapting Style with Llama..."):
                system_prompt = f"Rewrite this text to naturally sound like a {voice_option}. Keep output brief in English or Hindi script only, no symbols or commentary."
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": transcription}
                    ],
                    model="openai/gpt-oss-120b",
                )
                styled_text = chat_completion.choices[0].message.content.strip()
                st.success(f"**Adapted Script:** {styled_text}")

            # 3. Ultra-Realistic TTS Generation
            with st.spinner("Generating Natural AI Voice Output..."):
                output_audio_path = "natural_output.mp3"
                selected_voice = VOICE_MAPPING.get(voice_option, "hi-IN-SwaraNeural")

                # Run async TTS generator
                asyncio.run(generate_natural_voice(styled_text, selected_voice, output_audio_path))

                st.subheader("🎧 Realistic Human Voice Output:")
                st.audio(output_audio_path, format="audio/mp3")

        except Exception as e:
            st.error(f"Error: {str(e)}")
