"""
Enhanced synesthesia_ai.py with audio generation and additional features
"""

import os
import io
import traceback
import random
import time
from typing import List, Optional, Tuple
import numpy as np
from scipy.io.wavfile import write as write_wav
import sounddevice as sd
from pydub import AudioSegment
from pydub.generators import Sine, Square, Pulse, WhiteNoise

from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import login as hf_login
import torch
from gtts import gTTS

# ---------- Configuration helpers ----------
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if HUGGINGFACE_TOKEN:
    try:
        hf_login(HUGGINGFACE_TOKEN, add_to_git_credential=False)
    except Exception:
        pass

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
        else:
            wave = Sine(freq)
            
        return wave.to_audio_segment(duration=self.duration*1000).fade_out(self.fade_duration)
    
    def generate_sound_for_object(self, object_name: str) -> AudioSegment:
        """Generate a unique sound based on object characteristics"""
        # Map object to sound parameters
        sound_params = self._get_sound_parameters(object_name)
        
        # Generate multiple layers of sound
        layers = []
        for freq, wave_type, volume in sound_params:
            tone = self.generate_tone(freq, wave_type) - (20 - volume)
            layers.append(tone)
        
        # Mix layers with panning
        mixed = layers[0]
        for i, layer in enumerate(layers[1:]):
            pan_position = -0.5 + (i / (len(layers)-1))  # Distribute across stereo field
            mixed = mixed.overlay(layer.pan(pan_position))
            
        return mixed
    
    def _get_sound_parameters(self, object_name: str) -> List[Tuple[float, str, int]]:
        """Return frequency, wave type, and volume for sound layers"""
        # Hash object name to consistent parameters
        name_hash = hash(object_name.lower())
        random.seed(name_hash)
        
        # Base frequency based on object size category
        size_categories = {
            'small': (300, 800),
            'medium': (150, 400),
            'large': (50, 200)
        }
        size = random.choice(list(size_categories.keys()))
        base_freq = random.uniform(*size_categories[size])
        
        # Create 3-5 layers with related frequencies
        layers = []
        for i in range(random.randint(3, 5)):
            freq_variation = base_freq * (1 + 0.2 * i)
            wave_type = random.choice(["sine", "square", "pulse"])
            volume = random.randint(5, 15)
            layers.append((freq_variation, wave_type, volume))
            
        return layers

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
    """Generate poetic text with fallback"""
    if _poem_pipe:
        try:
            prompt = f"Create a short poetic line about: {text}"
            out = _poem_pipe(prompt, max_length=40, num_return_sequences=1)
            return out[0]["generated_text"].strip()
        except Exception:
            pass
    return f"A poetic thought about {text}."

def text_to_speech(text: str, output_path: str = "output.mp3") -> str:
    """Convert text to speech with fallback"""
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
