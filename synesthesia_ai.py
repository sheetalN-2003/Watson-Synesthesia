"""
Enhanced synesthesia_ai.py with IBM Watson integration and automatic IAM token generation
"""

import os
import io
import traceback
import random
import time"""
Robust Synesthesia AI implementation with IBM Watson integration
"""

import os
import io
import tempfile
import random
import time
import requests
import json
import logging
from typing import List, Optional, Tuple, Union
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta
from pydub import AudioSegment
from pydub.generators import Sine, Square, Pulse, WhiteNoise
from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import login as hf_login
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
        
        try:
            response = requests.post(self.iam_url, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            token_data = response.json()
            self.iam_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)
            return self.iam_token
        except Exception as e:
            logger.error(f"Failed to get IAM token: {str(e)}")
            raise

# Initialize services
try:
    ibm_authenticator = IBMCloudAuthenticator()
    USE_IBM_SERVICES = os.getenv("USE_IBM_SERVICES", "").lower() in ("1", "true", "yes")
    IBM_API_KEY = os.getenv("IBM_API_KEY")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    TTS_URL = os.getenv("TTS_URL", "https://api.us-south.text-to-speech.watson.cloud.ibm.com")
    TTS_VOICE = os.getenv("TTS_VOICE", "en-US_AllisonV3Voice")
except Exception as e:
    logger.error(f"Initialization error: {str(e)}")
    USE_IBM_SERVICES = False

# Initialize Hugging Face
try:
    if HUGGINGFACE_TOKEN := os.getenv("HUGGINGFACE_TOKEN"):
        hf_login(token=HUGGINGFACE_TOKEN, add_to_git_credential=False)
except Exception:
    logger.warning("Failed to login to Hugging Face Hub")

# ---------- Model Initialization ----------
def initialize_model(model_fn, model_name, **kwargs):
    """Safe model initialization with retries"""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            return model_fn(model_name, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
                return None
            time.sleep(2 ** attempt)

# Models with fallbacks
_caption_processor = initialize_model(BlipProcessor.from_pretrained, "Salesforce/blip-image-captioning-base")
_caption_model = initialize_model(BlipForConditionalGeneration.from_pretrained, "Salesforce/blip-image-captioning-base")

_emotion_pipe = initialize_model(
    pipeline, 
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
) or initialize_model(pipeline, "sentiment-analysis")

_poem_pipe = initialize_model(pipeline, "text-generation", model="gpt2")

try:
    _obj_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    logger.warning(f"Failed to load YOLOv5: {str(e)}")
    _obj_model = None

# ---------- Audio Generation ----------
class SynesthesiaSoundGenerator:
    def __init__(self):
        self.sample_rate = 44100
        self.duration = 1.5
        self.fade_duration = 200
        
    def generate_tone(self, freq: float, wave_type: str = "sine") -> AudioSegment:
        """Generate audio waveform"""
        wave_map = {
            "sine": Sine,
            "square": Square,
            "pulse": Pulse,
            "noise": WhiteNoise
        }
        generator = wave_map.get(wave_type, Sine)
        return generator(freq).to_audio_segment(
            duration=self.duration*1000
        ).fade_out(self.fade_duration)
    
    def generate_sound_for_object(self, object_name: str) -> AudioSegment:
        """Generate layered sound for an object"""
        params = self._get_sound_parameters(object_name.lower())
        base_audio = AudioSegment.silent(duration=self.duration*1000)
        
        for i, (freq, wave_type, volume) in enumerate(params):
            tone = self.generate_tone(freq, wave_type) - (20 - volume)
            pan = -0.5 + (i / max(1, len(params)-1))
            base_audio = base_audio.overlay(tone.pan(pan))
            
        return base_audio
    
    def _get_sound_parameters(self, object_name: str) -> List[Tuple[float, str, int]]:
        """Generate consistent sound parameters for an object"""
        random.seed(hash(object_name))
        size = random.choice(["small", "medium", "large"])
        freq_ranges = {"small": (300,800), "medium": (150,400), "large": (50,200)}
        base_freq = random.uniform(*freq_ranges[size])
        
        return [
            (
                base_freq * (1 + 0.2*i),
                random.choice(["sine", "square", "pulse"]),
                random.randint(5,15)
            )
            for i in range(random.randint(3,5))
        ]

sound_generator = SynesthesiaSoundGenerator()

# ---------- IBM Watson Services ----------
def watsonx_generate(prompt: str, model_id: str = "ibm/granite-13b-chat-v2") -> Optional[str]:
    """Generate text using Watsonx.ai"""
    if not (USE_IBM_SERVICES and IBM_API_KEY and WATSONX_PROJECT_ID):
        return None
        
    try:
        headers = {
            "Authorization": f"Bearer {ibm_authenticator.get_iam_token(IBM_API_KEY)}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{WATSONX_URL}/ml/v1-beta/generation/text?version=2023-05-28",
            headers=headers,
            json={
                "model_id": model_id,
                "input": prompt,
                "parameters": {"max_new_tokens": 100},
                "project_id": WATSONX_PROJECT_ID
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["results"][0]["generated_text"].strip()
    except Exception as e:
        logger.error(f"Watsonx generation failed: {str(e)}")
        return None

def watson_text_to_speech(text: str, output_path: str) -> bool:
    """Convert text to speech using Watson TTS"""
    if not (USE_IBM_SERVICES and IBM_API_KEY):
        return False
        
    try:
        response = requests.post(
            f"{TTS_URL}/v1/synthesize",
            headers={
                "Authorization": f"Bearer {ibm_authenticator.get_iam_token(IBM_API_KEY)}",
                "Content-Type": "application/json"
            },
            json={"text": text, "voice": TTS_VOICE},
            timeout=30
        )
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.error(f"Watson TTS failed: {str(e)}")
        return False

# ---------- Core Functions ----------
def caption_image(image: Image.Image) -> str:
    """Generate image caption with fallback"""
    if _caption_processor and _caption_model:
        try:
            inputs = _caption_processor(image, return_tensors="pt")
            outputs = _caption_model.generate(**inputs, max_new_tokens=40)
            return _caption_processor.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Captioning failed: {str(e)}")
    
    w, h = image.size
    return f"A {w}x{h} image with interesting content"

def detect_objects(image: Union[np.ndarray, Image.Image]) -> List[str]:
    """Detect objects in image with confidence threshold"""
    if not _obj_model:
        return []
    
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        results = _obj_model(image)
        return results.pandas().xyxy[0][results.pandas().xyxy[0]['confidence'] > 0.5]['name'].tolist()
    except Exception as e:
        logger.warning(f"Object detection failed: {str(e)}")
        return []

def analyze_emotion_text(text: str) -> str:
    """Analyze text emotion with fallback"""
    if not _emotion_pipe:
        return "neutral"
    
    try:
        result = _emotion_pipe(text)[0]
        label = result["label"].lower().replace("label_", "")
        return {"positive": "happy", "negative": "sad"}.get(label, label)
    except Exception:
        return "neutral"

def poetic_line(text: str) -> str:
    """Generate poetic description with fallbacks"""
    if USE_IBM_SERVICES:
        if watson_response := watsonx_generate(f"Create a poetic line about: {text}"):
            return watson_response
    
    if _poem_pipe:
        try:
            result = _poem_pipe(
                f"Create a poetic line about: {text}",
                max_length=50,
                num_return_sequences=1
            )
            return result[0]["generated_text"].strip()
        except Exception:
            pass
    
    return f"A poetic thought about {text}."

def text_to_speech(text: str, output_path: str) -> bool:
    """Convert text to speech with fallbacks"""
    if USE_IBM_SERVICES and watson_text_to_speech(text, output_path):
        return True
    
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(output_path)
        return True
    except Exception as e:
        logger.error(f"TTS failed: {str(e)}")
        try:
            AudioSegment.silent(duration=1000).export(output_path, format="mp3")
            return True
        except Exception:
            return False

# Object sound mappings
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
    """Get sound description for an object"""
    primary, secondary = OBJECT_SOUND_MAP.get(object_name.lower(), ("ambient tone", "harmonic"))
    return f"{primary} with hints of {secondary}"

def generate_audio_visualization(audio_path: str) -> Optional[BytesIO]:
    """Generate waveform visualization"""
    try:
        audio = AudioSegment.from_file(audio_path)
        samples = np.array(audio.get_array_of_samples())
        plt.figure(figsize=(10, 2), dpi=100)
        librosa.display.waveshow(samples.astype('float32'), sr=audio.frame_rate, color='#667eea')
        plt.axis('off')
        plt.margins(x=0)
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        img_bytes.seek(0)
        return img_bytes
    except Exception as e:
        logger.warning(f"Audio visualization failed: {str(e)}")
        return None
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
