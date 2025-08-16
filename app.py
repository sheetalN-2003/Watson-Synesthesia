import streamlit as st
import cv2
import numpy as np
from scipy.io.wavfile import write
import requests
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- App Title ---
st.title("🎨 Synesthesia AI: Real-Time Color Music")

# --- 1. Automatic IAM Token Generation ---
def get_iam_token(api_key):
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    response = requests.post(url, headers=headers, data=data)
    return response.json().get("access_token")

# --- 2. Initialize Watson Services ---
api_key = os.getenv("WATSON_API_KEY") or st.secrets.get("WATSON_API_KEY")
service_url = os.getenv("SERVICE_URL") or "https://api.us-south.visual-recognition.watson.cloud.ibm.com"

if not api_key:
    api_key = st.text_input("Enter IBM Cloud API Key", type="password")

if api_key:
    try:
        # Get fresh IAM token
        iam_token = get_iam_token(api_key)
        
        # Initialize Watson Visual Recognition
        authenticator = IAMAuthenticator(api_key)
        visual_recognition = VisualRecognitionV3(
            version='2018-03-19',
            authenticator=authenticator
        )
        visual_recognition.set_service_url(service_url)

        # --- 3. Image Processing Pipeline ---
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            # Process image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Your Image", use_column_width=True)
            
            # Extract dominant colors
            pixels = img_rgb.reshape(-1, 3)
            pixels = np.float32(pixels)
            _, labels, centers = cv2.kmeans(
                pixels, 3, None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                10, cv2.KMEANS_RANDOM_CENTERS
            )
            dominant_color = np.uint8(centers[np.bincount(labels.flatten()).argmax()])
            
            # --- 4. Dynamic Sound Generation ---
            color_map = {
                'red': (440, 'sine'),
                'green': (523.25, 'square'),
                'blue': (659.25, 'sawtooth')
            }
            
            # Determine closest color
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
            
            # --- 5. Watson Image Analysis ---
            with st.spinner("Analyzing image with Watson..."):
                # Save temp image for analysis
                with open("temp_img.jpg", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Get image analysis
                with open("temp_img.jpg", "rb") as images_file:
                    results = visual_recognition.classify(
                        images_file,
                        threshold='0.6',
                        classifier_ids='default').get_result()
                
                # Display results
                st.subheader("🔍 Image Analysis")
                classes = results['images'][0]['classifiers'][0]['classes']
                for obj in sorted(classes, key=lambda x: x['score'], reverse=True)[:3]:
                    st.write(f"{obj['class']} (confidence: {obj['score']:.0%})")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.warning("Please provide your IBM Cloud API key to continue")
