"""
app.py - Streamlit front-end for Watson Synesthesia POC (HuggingFace backend + gTTS)
Place this file at repo root alongside synesthesia_ai.py
"""

import os
import tempfile
from PIL import Image
import numpy as np
import streamlit as st

from synesthesia_ai import (
    caption_image,
    detect_objects,
    analyze_emotion_text,
    poetic_line,
    text_to_speech,
    object_to_sound_label,
)

st.set_page_config(page_title="AI Synesthesia (Accessibility POC)", layout="centered")
st.title("🎨 AI Synesthesia — Accessible Art for the Visually Challenged")

st.markdown(
    """
This demo:
- Camera mode: show an object to the camera → the app will detect it and speak a description + play a sound label.
- Image upload: upload an image → the app will caption it, create a short poem, and speak it.
- Text mode: enter a line about food or anything → the app will generate a poetic line and speak it.
"""
)

mode = st.radio("Mode", ["Camera (snapshot)", "Image Upload", "Text / Food"])

# ---------------- Camera mode ----------------
if mode == "Camera (snapshot)":
    st.subheader("Camera — show an object and take a snapshot")
    img_file = st.camera_input("Use your webcam / phone camera to take a photo")
    if img_file is not None:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Snapshot", use_column_width=True)
        # convert to numpy cv2 format
        frame = np.array(img)
        objects = detect_objects(frame)
        if objects:
            top = objects[0]
            st.success(f"I detect: {top}")
            # speak description
            poem = f"I see a {top}."
            # create audio
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            audio_path = tmp.name
            tmp.close()
            text_to_speech(poem, output_path=audio_path)
            st.audio(audio_path, format="audio/mp3")
            # show sound label
            label = object_to_sound_label(top)
            st.info(f"Sound palette for {top}: {label}")
        else:
            st.info("No objects confidently detected.")

# ---------------- Image Upload ----------------
elif mode == "Image Upload":
    st.subheader("Upload an image to create music + poetry")
    uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)
        with st.spinner("Generating caption..."):
            caption = caption_image(img)
        st.markdown(f"**Caption:** {caption}")

        with st.spinner("Generating poetic line..."):
            poem = poetic_line(caption)
        st.markdown(f"**Poem:** {poem}")

        # TTS
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        audio_path = tmp.name
        tmp.close()
        with st.spinner("Synthesizing speech..."):
            text_to_speech(poem, output_path=audio_path)
        st.audio(audio_path, format="audio/mp3")

# ---------------- Text mode ----------------
else:
    st.subheader("Write text (e.g. food description) and get poetic narration")
    txt = st.text_area("Write a line about a food, scene, or feeling", height=140)
    if st.button("Generate"):
        if not txt.strip():
            st.error("Please enter some text.")
        else:
            with st.spinner("Analyzing emotion..."):
                emotion = analyze_emotion_text(txt)
            with st.spinner("Creating poem..."):
                poem = poetic_line(txt)
            st.write(f"**Detected emotion:** {emotion}")
            st.write(f"**Poem:** {poem}")

            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            audio_path = tmp.name
            tmp.close()
            with st.spinner("Synthesizing speech..."):
                text_to_speech(poem, output_path=audio_path)
            st.audio(audio_path, format="audio/mp3")

# ---------------- Diagnostics / Tips ----------------
st.markdown("---")
st.write("**Notes / Troubleshooting**")
st.write(
    """
- If model downloads fail (Hugging Face rate limit), add a `HUGGINGFACE_TOKEN` to Streamlit Secrets.
- Object detection and BLIP are relatively heavy to download; patience required on first run.
- To use watsonx agentic features (optional), set environment variables:
  - USE_WATSONX=true
  - IBM_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL (optional)
  But watsonx is optional and not required to run this app.
"""
)
