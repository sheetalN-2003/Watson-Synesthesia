import streamlit as st
import cv2
import numpy as np
from scipy.io.wavfile import write
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- App Title ---
st.title("🎨 Watson Synesthesia: Color to Sound")

# --- Configuration ---
API_KEY = os.getenv("WATSON_API_KEY") or st.secrets.get("WATSON_API_KEY")
SERVICE_URL = "https://api.us-south.visual-recognition.watson.cloud.ibm.com/instances/YOUR_INSTANCE_ID"  # Replace with your instance ID

if not API_KEY:
    API_KEY = st.text_input("Enter IBM Cloud API Key", type="password")

if API_KEY and SERVICE_URL:
    try:
        # Initialize Watson Visual Recognition
        authenticator = IAMAuthenticator(API_KEY)
        visual_recognition = VisualRecognitionV3(
            version='2018-03-19',
            authenticator=authenticator
        )
        visual_recognition.set_service_url(SERVICE_URL)

        # --- Image Upload and Processing ---
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            # Process image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Your Image", use_container_width=True)  # Fixed deprecated parameter
            
            # --- Color Analysis ---
            pixels = img_rgb.reshape(-1, 3)
            pixels = np.float32(pixels)
            _, labels, centers = cv2.kmeans(
                pixels, 3, None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                10, cv2.KMEANS_RANDOM_CENTERS
            )
            dominant_color = np.uint8(centers[np.bincount(labels.flatten()).argmax()])
            
            # --- Sound Generation ---
            color_map = {
                'red': (440, 'sine'),
                'green': (523.25, 'square'),
                'blue': (659.25, 'sawtooth')
            }
            
            color_distances = {
                'red': np.linalg.norm(dominant_color - [255, 0, 0]),
                'green': np.linalg.norm(dominant_color - [0, 255, 0]),
                'blue': np.linalg.norm(dominant_color - [0, 0, 255])
            }
            closest_color = min(color_distances, key=color_distances.get)
            freq, wave_type = color_map[closest_color]
            
            # Generate audio
            duration = 2.0
            sr = 44100
            t = np.linspace(0, duration, int(sr * duration), False)
            
            if wave_type == 'sine':
                audio = np.sin(2 * np.pi * freq * t)
            elif wave_type == 'square':
                audio = np.sign(np.sin(2 * np.pi * freq * t))
            else:  # sawtooth
                audio = 2 * (t * freq - np.floor(0.5 + t * freq))
            
            audio = (audio * 32767 * 0.3 / np.max(np.abs(audio))).astype(np.int16)
            st.audio(audio, sample_rate=sr)
            
            # --- Watson Analysis ---
            with st.spinner("Analyzing with Watson..."):
                with open("temp_img.jpg", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    with open("temp_img.jpg", "rb") as images_file:
                        results = visual_recognition.classify(
                            images_file,
                            threshold='0.6'
                        ).get_result()
                    
                    st.subheader("🔍 Watson Analysis")
                    if 'images' in results and len(results['images']) > 0:
                        for obj in results['images'][0]['classifiers'][0]['classes'][:3]:
                            st.write(f"- {obj['class']} (confidence: {obj['score']:.0%})")
                    else:
                        st.warning("No analysis results returned")
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    except Exception as e:
        st.error(f"Service initialization error: {str(e)}")
else:
    st.warning("Please provide your IBM Cloud credentials")
