# app.py
import streamlit as st
from PIL import Image
import io
import os
from sonify import get_dominant_colors, rgb_to_freq, synthesize_tone_sequence
from ibm_client import nlu_analyze_text, tts_synthesize, get_iam_token_from_apikey
import base64

st.set_page_config(page_title="Watson Synesthesia POC", layout="centered")
st.title("Watson Synesthesia — POC")

# ---------------------------
# Configuration (from Streamlit secrets or env)
# ---------------------------
IBM_APIKEY = st.secrets.get("IBM_APIKEY", None) or os.environ.get("IBM_APIKEY")
NLU_URL = st.secrets.get("NLU_URL", None) or os.environ.get("NLU_URL")
TTS_URL = st.secrets.get("TTS_URL", None) or os.environ.get("TTS_URL")

if not IBM_APIKEY:
    st.warning("Set IBM_APIKEY in Streamlit secrets or environment to test NLU / TTS. Local sonification works without it.")
# ---------------------------

st.header("1) Color Sonification (upload or camera)")
img_file = st.file_uploader("Upload an image (or use camera below)", type=['png','jpg','jpeg'])
cam_img = st.camera_input("Or take a photo with your camera")

image = None
if cam_img:
    image = Image.open(cam_img)
elif img_file:
    image = Image.open(img_file)

if image is not None:
    st.image(image, caption="Input image", use_column_width=True)
    n = st.slider("Number of dominant colors", 3, 8, 5)
    colors = get_dominant_colors(image, n_colors=n)
    st.write("Dominant colors (RGB):", colors)
    freq_amp = [rgb_to_freq(c) for c in colors]
    audio_bytes = synthesize_tone_sequence(freq_amp, duration_per_tone=st.slider("Tone duration (s)", 0.2, 1.5, 0.6))
    st.audio(audio_bytes, format='audio/wav')
    # show color swatches
    cols = st.columns(len(colors))
    for c,col in zip(colors,cols):
        col.markdown(f"<div style='width:100%;height:60px;background:rgb{tuple(c)}'></div>", unsafe_allow_html=True)

st.markdown("---")
st.header("2) Text → Emotion → Flavor / Poetry (NLU + TTS)")
user_text = st.text_area("Paste a line / caption / review to translate into 'flavor poetry' or emotions", height=120)
if st.button("Analyze & Generate"):
    if not IBM_APIKEY or not NLU_URL or not TTS_URL:
        st.error("NLU/TTS credentials (IBM_APIKEY, NLU_URL, TTS_URL) required for this block. Sonification above works offline.")
    else:
        with st.spinner("Analyzing text for emotion..."):
            nlu_resp = nlu_analyze_text(IBM_APIKEY, NLU_URL, user_text)
            emotions = nlu_resp.get("emotion", {}).get("document", {}).get("emotion", {})
        st.write("Emotion scores:", emotions)
        # map top emotion to flavor / poetry heuristic
        if emotions:
            top = max(emotions.items(), key=lambda x: x[1])[0]
            mapping = {
                "joy": "sweet citrus, honeyed breeze",
                "sadness": "deep umami, slow-brewed tea",
                "anger": "fiery chili, black pepper crack",
                "fear": "sour tang, metallic fizz",
                "disgust": "bitter rind, chalky ash"
            }
            flavor = mapping.get(top, "complex spice")
            poem = f"{user_text}\n→ Emotion: {top}  —  Flavor: {flavor}\nHaiku:\n{user_text[:20]} / {flavor} / echoes"
            st.markdown(f"**Top emotion:** {top}  \n**Flavor-poem:** {poem}")
            # synthesize TTS
            with st.spinner("Generating speech..."):
                try:
                    audio = tts_synthesize(IBM_APIKEY, TTS_URL, poem)
                    st.audio(audio, format='audio/wav')
                except Exception as e:
                    st.error(f"TTS failed: {e}")

st.markdown("---")
st.write("## Notes & Next steps")
st.write("""
- This POC uses local image color extraction + waveform synthesis for instant demos (works offline).
- For IBM-based vision/emotion mapping in production: use IBM NLU for text emotion; visual emotion likely needs a vision model (watsonx or custom model), since Watson Visual Recognition was discontinued.
""")
