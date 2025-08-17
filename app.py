"""
Streamlit application for Synesthesia Experience
"""

import os
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
from pydub import AudioSegment
from synesthesia_ai import (
    caption_image,
    detect_objects,
    analyze_emotion_text,
    poetic_line,
    text_to_speech,
    describe_object_sound,
    generate_object_sound,
    generate_audio_visualization,
    USE_IBM_SERVICES
)

# Configure page
st.set_page_config(
    page_title="Synesthesia Experience",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stRadio > div {
        background-color: rgba(255,255,255,0.8);
        padding: 10px;
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .ibm-badge {
        background-color: #0062FF;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .highlight {
        background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(102,126,234,0.2) 100%);
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("🎨 Synesthesia Experience")
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <p style="font-size: 18px;">
        Translate visual input into rich audio experiences
        {" " + "<span class='ibm-badge'>IBM Watson</span>" if USE_IBM_SERVICES else ""}
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Experience Mode",
        ["Camera Snapshot", "Image Upload", "Text Input", "Sound Explorer"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("Audio Preferences")
    voice_speed = st.slider("Voice Speed", 0.5, 2.0, 1.0, 0.1)
    sound_duration = st.slider("Sound Duration (sec)", 0.5, 3.0, 1.5, 0.1)
    
    st.markdown("---")
    st.info("""
    **Tips:**
    - For best results, use clear images with one main subject
    - Try different objects to hear unique sound signatures
    - Explore the Sound Explorer mode to customize audio
    """)

# Helper function for audio processing
def create_audio_file(audio_segment: AudioSegment) -> str:
    """Create temporary audio file from segment"""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_segment.export(tmp.name, format="mp3")
        return tmp.name

# Main app logic
if mode == "Camera Snapshot":
    st.subheader("📸 Camera Experience")
    img_file = st.camera_input("Take a photo")
    
    if img_file:
        img = Image.open(img_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, use_column_width=True)
            objects = detect_objects(img)
            
            if objects:
                obj = objects[0]
                st.success(f"Detected: {obj.capitalize()}")
                st.markdown(f"**Sound:** {describe_object_sound(obj)}")
                
                # Generate and play sound
                sound = generate_object_sound(obj)
                sound_file = create_audio_file(sound)
                st.audio(sound_file, format="audio/mp3")
                
                # Generate description
                poem = poetic_line(f"A {obj}")
                st.markdown(f"**Description:** {poem}")
                
                # Generate voice
                voice_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
                if text_to_speech(poem, voice_file):
                    st.audio(voice_file, format="audio/mp3")
            else:
                st.info("No objects detected")

elif mode == "Image Upload":
    st.subheader("🖼️ Image Experience")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, use_column_width=True)
            
        with col2:
            caption = caption_image(img)
            st.markdown(f"**Caption:** {caption}")
            
            poem = poetic_line(caption)
            st.markdown(f"**Poetic Description:** {poem}")
            
            # Generate combined audio
            voice_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            text_to_speech(poem, voice_file)
            
            objects = detect_objects(img)[:3]
            if objects:
                soundscape = AudioSegment.silent(duration=1000)
                for obj in objects:
                    sound = generate_object_sound(obj)
                    soundscape = soundscape.overlay(sound)
                
                combined = AudioSegment.from_file(voice_file).overlay(soundscape)
                combined_file = create_audio_file(combined)
                st.audio(combined_file, format="audio/mp3")
            else:
                st.audio(voice_file, format="audio/mp3")

elif mode == "Text Input":
    st.subheader("✍️ Text Experience")
    text = st.text_area("Enter text to describe")
    
    if st.button("Generate") and text.strip():
        col1, col2 = st.columns(2)
        
        with col1:
            emotion = analyze_emotion_text(text)
            st.markdown(f"**Emotion:** {emotion.capitalize()}")
            
            poem = poetic_line(text)
            st.markdown(f"**Description:** {poem}")
            
            voice_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            if text_to_speech(poem, voice_file):
                st.audio(voice_file, format="audio/mp3")
        
        with col2:
            sound = generate_object_sound(text)
            sound_file = create_audio_file(sound)
            
            if viz := generate_audio_visualization(sound_file):
                st.image(viz, use_column_width=True)
            st.audio(sound_file, format="audio/mp3")

else:  # Sound Explorer
    st.subheader("🎵 Sound Explorer")
    
    col1, col2 = st.columns(2)
    with col1:
        wave_type = st.selectbox("Wave Type", ["sine", "square", "pulse", "noise"])
        freq = st.slider("Frequency (Hz)", 50, 1000, 220)
        duration = st.slider("Duration (s)", 0.5, 3.0, 1.5)
    
    with col2:
        layers = st.slider("Layers", 1, 5, 2)
        detune = st.slider("Detune (%)", 0, 50, 10)
        volume = st.slider("Volume", 0, 20, 10)
    
    if st.button("Generate Sound"):
        base_audio = AudioSegment.silent(duration=int(duration*1000))
        for i in range(layers):
            layer_freq = freq * (1 + (i * detune/100))
            layer = AudioSegment.silent(duration=int(duration*1000)).overlay(
                getattr(AudioSegment, wave_type)(layer_freq).to_audio_segment(
                    duration=int(duration*1000)
                ) - (20 - volume)
            )
            pan = -0.5 + (i / max(1, layers-1))
            base_audio = base_audio.overlay(layer.pan(pan))
        
        sound_file = create_audio_file(base_audio)
        if viz := generate_audio_visualization(sound_file):
            st.image(viz, use_column_width=True)
        st.audio(sound_file, format="audio/mp3")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px;">
    <p>Synesthesia Experience | Combining AI models for multi-sensory experiences</p>
</div>
""", unsafe_allow_html=True)
