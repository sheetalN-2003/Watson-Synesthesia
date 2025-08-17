"""
Robust Synesthesia AI implementation with proper error handling
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

from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import login as hf_login
import torch
from gtts import gTTS
from pydub import AudioSegment
from pydub.generators import Sine, Square, Pulse, WhiteNoise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models with error handling
try:
    # Initialize BLIP for image captioning
    _caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    _caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
except Exception as e:
    logger.warning(f"Failed to load BLIP models: {str(e)}")
    _caption_processor = None
    _caption_model = None

try:
    # Initialize emotion classification pipeline
    _emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
except Exception:
    try:
        _emotion_pipe = pipeline("sentiment-analysis")
    except Exception:
        _emotion_pipe = None
        logger.warning("Failed to load emotion analysis model")

try:
    # Initialize YOLOv5 for object detection
    _obj_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    logger.warning(f"Failed to load YOLOv5: {str(e)}")
    _obj_model = None

try:
    # Initialize text generation pipeline
    _poem_pipe = pipeline("text-generation", model="gpt2")
except Exception as e:
    logger.warning(f"Failed to load text generation model: {str(e)}")
    _poem_pipe = None

# Audio Generator Class
class SynesthesiaSoundGenerator:
    def __init__(self):
        self.sample_rate = 44100
        self.duration = 1.5
        self.fade_duration = 200
        
    def generate_tone(self, freq: float, wave_type: str = "sine") -> AudioSegment:
        try:
            if wave_type == "sine":
                wave = Sine(freq)
            elif wave_type == "square":
                wave = Square(freq)
            elif wave_type == "pulse":
                wave = Pulse(freq)
            else:
                wave = Sine(freq)
            return wave.to_audio_segment(duration=self.duration*1000).fade_out(self.fade_duration)
        except Exception as e:
            logger.error(f"Failed to generate tone: {str(e)}")
            return AudioSegment.silent(duration=self.duration*1000)
    
    def generate_sound_for_object(self, object_name: str) -> AudioSegment:
        try:
            random.seed(hash(object_name.lower()))
            base_freq = random.uniform(100, 800)
            layers = []
            
            for i in range(random.randint(2, 4)):
                freq = base_freq * (1 + 0.2 * i)
                wave_type = random.choice(["sine", "square", "pulse"])
                volume = random.randint(5, 15)
                layers.append((freq, wave_type, volume))
            
            base_audio = AudioSegment.silent(duration=self.duration*1000)
            for i, (freq, wave_type, volume) in enumerate(layers):
                tone = self.generate_tone(freq, wave_type) - (20 - volume)
                pan = -0.5 + (i / max(1, len(layers)-1))
                base_audio = base_audio.overlay(tone.pan(pan))
            
            return base_audio
        except Exception as e:
            logger.error(f"Failed to generate object sound: {str(e)}")
            return AudioSegment.silent(duration=self.duration*1000)

sound_generator = SynesthesiaSoundGenerator()

# Core Functions
def caption_image(image: Image.Image) -> str:
    try:
        if _caption_processor and _caption_model:
            inputs = _caption_processor(image, return_tensors="pt")
            outputs = _caption_model.generate(**inputs, max_new_tokens=40)
            return _caption_processor.decode(outputs[0], skip_special_tokens=True)
        return f"An image of size {image.size[0]}x{image.size[1]}"
    except Exception as e:
        logger.error(f"Captioning failed: {str(e)}")
        return "An interesting image"

def detect_objects(image: Union[np.ndarray, Image.Image]) -> List[str]:
    try:
        if _obj_model is None:
            return []
        if isinstance(image, Image.Image):
            image = np.array(image)
        results = _obj_model(image)
        return results.pandas().xyxy[0][results.pandas().xyxy[0]['confidence'] > 0.5]['name'].tolist()
    except Exception as e:
        logger.error(f"Object detection failed: {str(e)}")
        return []

def analyze_emotion_text(text: str) -> str:
    try:
        if _emotion_pipe is None:
            return "neutral"
        result = _emotion_pipe(text)[0]
        label = result["label"].lower()
        return {"positive": "happy", "negative": "sad"}.get(label, label)
    except Exception as e:
        logger.error(f"Emotion analysis failed: {str(e)}")
        return "neutral"

def poetic_line(text: str) -> str:
    try:
        if _poem_pipe:
            result = _poem_pipe(f"Create a poetic line about: {text}", max_length=50, num_return_sequences=1)
            return result[0]["generated_text"].strip()
        return f"A poetic thought about {text}."
    except Exception as e:
        logger.error(f"Poetic generation failed: {str(e)}")
        return f"An artistic interpretation of {text}."

def text_to_speech(text: str, output_path: str) -> bool:
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

def describe_object_sound(object_name: str) -> str:
    sound_map = {
        "person": ("human voice", "choir"),
        "bicycle": ("bells", "click"),
        "car": ("engine rumble", "horn"),
        "dog": ("bark", "whine"),
        "cat": ("purr", "meow"),
        "bird": ("chirp", "song"),
    }
    primary, secondary = sound_map.get(object_name.lower(), ("ambient tone", "harmonic"))
    return f"{primary} with hints of {secondary}"

def generate_audio_visualization(audio_path: str) -> Optional[BytesIO]:
    try:
        audio = AudioSegment.from_file(audio_path)
        samples = np.array(audio.get_array_of_samples())
        plt.figure(figsize=(10, 2))
        librosa.display.waveshow(samples.astype('float32'), sr=audio.frame_rate)
        plt.axis('off')
        plt.margins(x=0)
        
        img_bytes = BytesIO()
        plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        img_bytes.seek(0)
        return img_bytes
    except Exception as e:
        logger.error(f"Audio visualization failed: {str(e)}")
        return None
