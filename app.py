import streamlit as st
import cv2
import numpy as np
from scipy.io.wavfile import write
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os

# --- Set Up Watson ---
authenticator = IAMAuthenticator(os.getenv('WATSON_API_KEY'))  # From Render env vars
visual_recognition = VisualRecognitionV3(
    version='2018-03-19',
    authenticator=authenticator
)
visual_recognition.set_service_url(os.getenv('WATSON_SERVICE_URL'))

# --- Streamlit UI ---
st.title("🎨 Watson Synesthesia: Color → Music")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    # --- Process Image ---
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Your Image", use_column_width=True)

    # --- Watson Color Analysis ---
    with open("temp_img.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    with open("temp_img.jpg", "rb") as img_file:
        classes = visual_recognition.classify(img_file).get_result()
    
    # Extract dominant color (simplified)
    dominant_color = classes['images'][0]['colors'][0]['hex']
    r, g, b = int(dominant_color[1:3], int(dominant_color[3:5]), int(dominant_color[5:7])

    # --- Color → Music Mapping ---
    note_freq = {
        'red': 440,    # A4
        'green': 523,  # C5
        'blue': 659    # E5
    }
    freq = note_freq['red'] if r > max(g, b) else note_freq['green'] if g > b else note_freq['blue']

    # --- Generate Sound ---
    duration = 2.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * freq * t)
    audio = (audio * 32767 * 0.3).astype(np.int16)  # Reduce volume

    # --- Play Audio ---
    st.audio(audio, sample_rate=sample_rate)
    st.json(classes)  # Show Watson's analysis
