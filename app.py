"""
Enhanced Streamlit app with audio visualization and more interactive features
"""

import os
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from synesthesia_ai import (
    caption_image,
    detect_objects,
    analyze_emotion_text,
    poetic_line,
    text_to_speech,
    describe_object_sound,
    generate_object_sound,
)

# App configuration
st.set_page_config(
    page_title="AI Synesthesia Experience",
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
    .stFileUploader>div>div>div>div {
        color: #764ba2;
    }
    .highlight {
        background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(102,126,234,0.2) 100%);
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🎨 AI Synesthesia Experience")
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <p style="font-size: 18px;">An immersive experience that translates visual input into rich audio descriptions and unique soundscapes</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with settings
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

# Audio visualization function
def plot_audio_waveform(audio_path):
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        samples = np.array(audio.get_array_of_samples())
        
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(samples, color='#667eea')
        ax.axis('off')
        ax.margins(x=0)
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        img_bytes.seek(0)
        return img_bytes
    except Exception:
        return None

# Main app logic
if mode == "Camera Snapshot":
    st.subheader("📸 Camera Experience")
    st.markdown("Show an object to your camera and experience its sound signature")
    
    img_file = st.camera_input("Take a photo with your camera")
    if img_file is not None:
        img = Image.open(img_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Your Image", use_column_width=True)
            
            with st.spinner("Detecting objects..."):
                frame = np.array(img)
                objects = detect_objects(frame)
                
            if objects:
                top_object = objects[0]
                st.success(f"Primary object detected: **{top_object.capitalize()}**")
                
                # Generate and play sound
                with st.spinner("Creating soundscape..."):
                    sound_desc = describe_object_sound(top_object)
                    st.markdown(f"**Sound palette:** {sound_desc}")
                    
                    tmp_sound = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    generate_object_sound(top_object, tmp_sound.name)
                    
                    # Audio visualization
                    waveform = plot_audio_waveform(tmp_sound.name)
                    if waveform:
                        st.image(waveform, use_column_width=True)
                    
                    st.audio(tmp_sound.name, format="audio/mp3")
                
                # Generate description
                with st.spinner("Creating description..."):
                    poem = poetic_line(f"A {top_object}")
                    st.markdown(f"**Poetic impression:** {poem}")
                    
                    tmp_voice = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    text_to_speech(poem, tmp_voice.name)
                    st.audio(tmp_voice.name, format="audio/mp3")
            else:
                st.info("No objects confidently detected. Try getting closer to your subject.")

elif mode == "Image Upload":
    st.subheader("🖼️ Image Experience")
    st.markdown("Upload an image to explore its audio representation")
    
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
        with col2:
            with st.spinner("Analyzing image..."):
                caption = caption_image(img)
                st.markdown(f"**Image caption:** {caption}")
                
                poem = poetic_line(caption)
                st.markdown(f"**Poetic interpretation:** {poem}")
                
                # Generate audio for all detected objects
                frame = np.array(img)
                objects = detect_objects(frame)
                
                if objects:
                    st.markdown("**Detected objects:**")
                    for obj in objects[:3]:  # Limit to top 3 objects
                        sound_desc = describe_object_sound(obj)
                        st.markdown(f"- {obj.capitalize()}: {sound_desc}", unsafe_allow_html=True)
                
                # Create combined audio experience
                tmp_combined = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                
                # Generate voice description
                tmp_voice = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                text_to_speech(poem, tmp_voice.name)
                voice_audio = AudioSegment.from_file(tmp_voice.name)
                
                # Generate soundscape
                if objects:
                    soundscape = AudioSegment.silent(duration=1000)  # 1s silence
                    for obj in objects[:3]:  # Limit to 3 objects
                        tmp_sound = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                        generate_object_sound(obj, tmp_sound.name)
                        obj_audio = AudioSegment.from_file(tmp_sound.name)
                        soundscape = soundscape.overlay(obj_audio)
                    
                    # Combine voice and soundscape
                    combined = voice_audio.overlay(soundscape)
                    combined.export(tmp_combined.name, format="mp3")
                else:
                    voice_audio.export(tmp_combined.name, format="mp3")
                
                # Show waveform
                waveform = plot_audio_waveform(tmp_combined.name)
                if waveform:
                    st.image(waveform, use_column_width=True)
                
                st.audio(tmp_combined.name, format="audio/mp3")

elif mode == "Text Input":
    st.subheader("✍️ Text Experience")
    st.markdown("Describe something and hear its sound representation")
    
    txt = st.text_area("Describe an object, scene, or feeling", height=100)
    if st.button("Generate Experience"):
        if txt.strip():
            with st.spinner("Creating your experience..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    emotion = analyze_emotion_text(txt)
                    st.markdown(f"**Emotion detected:** {emotion.capitalize()}")
                    
                    poem = poetic_line(txt)
                    st.markdown(f"**Poetic interpretation:** {poem}")
                
                with col2:
                    # Generate sound based on text
                    tmp_sound = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    generate_object_sound(txt, tmp_sound.name)
                    
                    waveform = plot_audio_waveform(tmp_sound.name)
                    if waveform:
                        st.image(waveform, use_column_width=True)
                    
                    st.audio(tmp_sound.name, format="audio/mp3")
                    
                    # Generate voice
                    tmp_voice = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    text_to_speech(poem, tmp_voice.name)
                    st.audio(tmp_voice.name, format="audio/mp3")
        else:
            st.warning("Please enter some text to generate an experience")

else:  # Sound Explorer
    st.subheader("🎵 Sound Explorer")
    st.markdown("Experiment with different sound parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        wave_type = st.selectbox("Waveform Type", ["sine", "square", "pulse", "noise"])
        freq = st.slider("Base Frequency (Hz)", 50, 1000, 220, 10)
        duration = st.slider("Duration (seconds)", 0.5, 3.0, 1.5, 0.1)
        
    with col2:
        layers = st.slider("Number of Layers", 1, 5, 2)
        detune = st.slider("Detune Amount (%)", 0, 50, 10)
        volume = st.slider("Volume", 0, 20, 10)
    
    if st.button("Generate Sound"):
        with st.spinner("Creating sound..."):
            tmp_sound = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            
            # Generate multiple layers
            base_audio = AudioSegment.silent(duration=0)
            for i in range(layers):
                layer_freq = freq * (1 + (i * detune/100))
                layer = sound_generator.generate_tone(layer_freq, wave_type) - (20 - volume)
                base_audio = base_audio.overlay(layer.pan(-0.5 + (i/(layers-1)) if layers > 1 else 0))
            
            base_audio.export(tmp_sound.name, format="mp3")
            
            waveform = plot_audio_waveform(tmp_sound.name)
            if waveform:
                st.image(waveform, use_column_width=True)
            
            st.audio(tmp_sound.name, format="audio/mp3")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px;">
    <p>AI Synesthesia Experience | Combining computer vision, NLP, and audio synthesis</p>
</div>
""", unsafe_allow_html=True)
