"""
Enhanced synesthesia_ai.py with IBM Watson integration and automatic IAM token generation
"""

import os
import io
import traceback
import random
import time
import requests
import json
from typing import List, Optional, Tuple
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime, timedelta

from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import login as hf_login
import torch
from gtts import gTTS
from pydub import AudioSegment
from pydub.generators import Sine, Square, Pulse, WhiteNoise

# ---------- IBM Cloud Configuration ----------
class IBMCloudAuthenticator:
    def __init__(self):
        self.iam_token = None
        self.token_expiry = None
        self.iam_url = "https://iam.cloud.ibm.com/identity/token"
        
    def get_iam_token(self, api_key: str) -> str:
        """Get or refresh IAM token automatically"""
        if self.iam_token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.iam_token
            
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key
        }
        
        try:
            response = requests.post(self.iam_url, headers=headers, data=data)
            response.raise_for_status()
            token_data = response.json()
            self.iam_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)  # Refresh 5 min early
            return self.iam_token
        except Exception as e:
            raise Exception(f"Failed to get IAM token: {str(e)}")

# Initialize IBM Cloud services if configured
ibm_authenticator = IBMCloudAuthenticator()
USE_IBM_SERVICES = os.environ.get("USE_IBM_SERVICES", "false").lower() in ("1", "true", "yes")
IBM_API_KEY = os.environ.get("IBM_API_KEY")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
WATSONX_URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
TTS_URL = os.environ.get("TTS_URL", "https://api.us-south.text-to-speech.watson.cloud.ibm.com")
TTS_VOICE = os.environ.get("TTS_VOICE", "en-US_AllisonV3Voice")

# ---------- Model loaders with fallbacks ----------
def load_model_with_fallback(model_fn, fallback_fn, *args, **kwargs):
    try:
        return model_fn(*args, **kwargs)
    except Exception:
        return fallback_fn()

# Captioning (BLIP)
_caption_processor, _caption_model = None, None
try:
    _caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    _caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
except Exception:
    _caption_processor = None
    _caption_model = None

# Emotion classifier
_emotion_pipe = load_model_with_fallback(
    lambda: pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base"),
    lambda: pipeline("sentiment-analysis")
)

# Generator for poetic lines
_poem_pipe = load_model_with_fallback(
    lambda: pipeline("text-generation", model="gpt2"),
    lambda: None
)

# Object detection (yolov5 via torch.hub)
_obj_model = load_model_with_fallback(
    lambda: torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True),
    lambda: None
)

# ---------- IBM Watson Services ----------
def watsonx_generate(prompt: str, model_id: str = "ibm/granite-13b-chat-v2") -> str:
    """Generate text using Watsonx.ai"""
    if not USE_IBM_SERVICES or not IBM_API_KEY:
        return None
        
    try:
        iam_token = ibm_authenticator.get_iam_token(IBM_API_KEY)
        headers = {
            "Authorization": f"Bearer {iam_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "model_id": model_id,
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 100,
                "min_new_tokens": 10,
                "repetition_penalty": 1.0
            },
            "project_id": WATSONX_PROJECT_ID
        }
        
        response = requests.post(
            f"{WATSONX_URL}/ml/v1-beta/generation/text?version=2023-05-28",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["results"][0]["generated_text"].strip()
    except Exception as e:
        print(f"Watsonx generation failed: {str(e)}")
        return None

def watson_text_to_speech(text: str, output_path: str) -> bool:
    """Convert text to speech using Watson TTS"""
    if not USE_IBM_SERVICES or not IBM_API_KEY:
        return False
        
    try:
        iam_token = ibm_authenticator.get_iam_token(IBM_API_KEY)
        headers = {
            "Authorization": f"Bearer {iam_token}",
            "Content-Type": "application/json",
            "Accept": "audio/mp3"
        }
        
        payload = {
            "text": text,
            "voice": TTS_VOICE
        }
        
        response = requests.post(
            f"{TTS_URL}/v1/synthesize",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Watson TTS failed: {str(e)}")
        return False

# ---------- Audio Generation ----------
class SynesthesiaSoundGenerator:
    def __init__(self):
        self.sample_rate = 44100
        self.duration = 1.5  # seconds
        self.fade_duration = 200  # ms
        
    def generate_tone(self, freq: float, wave_type: str = "sine") -> AudioSegment:
        """Generate different waveform types"""
        if wave_type == "sine":
            wave = Sine(freq)
        elif wave_type == "square":
            wave = Square(freq)
        elif wave_type == "pulse":
            wave = Pulse(freq)
        elif wave_type == "noise":
            wave = WhiteNoise()
        else:
            wave = Sine(freq)
            
        return wave.to_audio_segment(duration=self.duration*1000).fade_out(self.fade_duration)
    
    def generate_sound_for_object(self, object_name: str) -> AudioSegment:
        """Generate a unique sound based on object characteristics"""
        sound_params = self._get_sound_parameters(object_name)
        
        # Generate multiple layers of sound
        base_audio = AudioSegment.silent(duration=self.duration*1000)
        for i, (freq, wave_type, volume) in enumerate(sound_params):
            tone = self.generate_tone(freq, wave_type) - (20 - volume)
            pan_position = -0.5 + (i / (len(sound_params)-1)) if len(sound_params) > 1 else 0
            base_audio = base_audio.overlay(tone.pan(pan_position))
            
        return base_audio
    
    def _get_sound_parameters(self, object_name: str) -> List[Tuple[float, str, int]]:
        """Return frequency, wave type, and volume for sound layers"""
        name_hash = hash(object_name.lower())
        random.seed(name_hash)
        
        size_categories = {
            'small': (300, 800),
            'medium': (150, 400),
            'large': (50, 200)
        }
        size = random.choice(list(size_categories.keys()))
        base_freq = random.uniform(*size_categories[size])
        
        layers = []
        for i in range(random.randint(3, 5)):
            freq_variation = base_freq * (1 + 0.2 * i)
            wave_type = random.choice(["sine", "square", "pulse"])
            volume = random.randint(5, 15)
            layers.append((freq_variation, wave_type, volume))
            
        return layers

    def generate_audio_visualization(self, audio_segment: AudioSegment) -> BytesIO:
        """Generate waveform visualization using librosa"""
        samples = np.array(audio_segment.get_array_of_samples())
        sr = audio_segment.frame_rate
        
        plt.figure(figsize=(10, 2), dpi=100)
        librosa.display.waveshow(samples.astype('float32'), sr=sr, color='#667eea')
        plt.axis('off')
        plt.margins(x=0)
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        img_bytes.seek(0)
        return img_bytes

sound_generator = SynesthesiaSoundGenerator()

# ---------- Public helper functions ----------
def caption_image(image: Image.Image) -> str:
    """Return a caption for a PIL image with fallback"""
    if _caption_processor and _caption_model:
        try:
            inputs = _caption_processor(images=image, return_tensors="pt")
            out = _caption_model.generate(**inputs, max_new_tokens=40)
            return _caption_processor.decode(out[0], skip_special_tokens=True)
        except Exception:
            pass
    
    # Fallback heuristic
    w, h = image.size
    return f"A {w}x{h} image with interesting content"

def detect_objects(frame) -> List[str]:
    """Detect objects with confidence threshold"""
    if _obj_model is None:
        return []
    
    try:
        results = _obj_model(frame)
        df = results.pandas().xyxy[0]
        confident_detections = df[df['confidence'] > 0.5]['name'].tolist()
        return confident_detections
    except Exception:
        return []

def analyze_emotion_text(text: str) -> str:
    """Analyze text emotion with fallback"""
    if not _emotion_pipe:
        return "neutral"
    
    try:
        res = _emotion_pipe(text)
        label = res[0].get("label", "").lower()
        emotion_map = {
            'positive': 'happy',
            'negative': 'sad',
            'neutral': 'neutral',
            'joy': 'happy',
            'anger': 'angry',
            'sadness': 'sad',
            'fear': 'fearful'
        }
        return emotion_map.get(label.split('_')[-1], label.split('_')[-1])
    except Exception:
        return "neutral"

def poetic_line(text: str) -> str:
    """Generate poetic text with IBM Watson fallback to local model"""
    if USE_IBM_SERVICES:
        watson_response = watsonx_generate(f"Create a short poetic line about: {text}")
        if watson_response:
            return watson_response
    
    if _poem_pipe:
        try:
            prompt = f"Create a short poetic line about: {text}"
            out = _poem_pipe(prompt, max_length=40, num_return_sequences=1)
            return out[0]["generated_text"].strip()
        except Exception:
            pass
    return f"A poetic thought about {text}."

def text_to_speech(text: str, output_path: str = "output.mp3") -> str:
    """Convert text to speech with IBM Watson fallback to gTTS"""
    if USE_IBM_SERVICES:
        if watson_text_to_speech(text, output_path):
            return output_path
    
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(output_path)
        return output_path
    except Exception as e:
        # Generate a simple tone as fallback
        audio = sound_generator.generate_tone(220)
        audio.export(output_path, format="mp3")
        return output_path

def generate_object_sound(object_name: str, output_path: str = None) -> Optional[AudioSegment]:
    """Generate and optionally save a unique sound for an object"""
    sound = sound_generator.generate_sound_for_object(object_name)
    if output_path:
        sound.export(output_path, format="mp3")
    return sound

def plot_audio_waveform(audio_path: str) -> BytesIO:
    """Generate waveform visualization for an audio file"""
    audio = AudioSegment.from_file(audio_path)
    return sound_generator.generate_audio_visualization(audio)

# Enhanced object to sound mapping
OBJECT_SOUND_MAP = {
    "person": ("human voice", "choir"),
    "bicycle": ("bells", "click"),
    "car": ("engine rumble", "horn"),
    "dog": ("bark", "whine"),
    "cat": ("purr", "meow"),
    "bird": ("chirp", "song"),
    "cup": ("cling", "pour"),
    "chair": ("creak", "wood knock"),
    "laptop": ("keyboard", "beep"),
    "book": ("page turn", "writing"),
    "phone": ("ringtone", "notification"),
}

def describe_object_sound(object_name: str) -> str:
    """Return a rich description of the sound palette"""
    primary, secondary = OBJECT_SOUND_MAP.get(object_name.lower(), ("ambient tone", "harmonic"))
    return f"{primary} with hints of {secondary}"
