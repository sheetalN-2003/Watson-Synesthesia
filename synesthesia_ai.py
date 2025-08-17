"""
synesthesia_ai.py
Utility functions for AI Synesthesia app:
- image captioning (BLIP)
- object detection (yolov5)
- emotion classification (HF)
- poetic generation (gpt2 fallback)
- text-to-speech (gTTS)
This file uses Hugging Face transformers. It will try to load models but provides
safe fallbacks if model downloads fail (useful on limited-host environments).
"""

import os
import io
import traceback
from typing import List, Optional

from PIL import Image

# Transformers + HF hub
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import login as hf_login

# PyTorch hub for YOLO (yolov5)
import torch

# gTTS for TTS (cloud-safe)
from gtts import gTTS

# ---------- Configuration helpers ----------
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if HUGGINGFACE_TOKEN:
    try:
        hf_login(HUGGINGFACE_TOKEN, add_to_git_credential=False)
    except Exception:
        # ignore login errors; model downloads may still work with public models
        pass

USE_WATSONX = os.environ.get("USE_WATSONX", "false").lower() in ("1", "true", "yes")
# If you want to enable watsonx, set USE_WATSONX=true and provide IBM_APIKEY & PROJECT_ID.
# Code will attempt to import IBM client only then.
if USE_WATSONX:
    try:
        from ibm_watsonx_ai import APIClient
        IBM_APIKEY = os.getenv("IBM_APIKEY")
        PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
        WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        credentials = {"url": WATSONX_URL, "apikey": IBM_APIKEY}
        client = APIClient(credentials)
        if PROJECT_ID:
            client.set.default_project(PROJECT_ID)
        else:
            client = None
    except Exception:
        client = None
else:
    client = None

# ---------- Model loaders with fallbacks ----------
# Captioning (BLIP)
_caption_processor = None
_caption_model = None
try:
    _caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    _caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
except Exception as e:
    # model download may fail on restricted envs; fallback will be a simple placeholder caption
    _caption_processor = None
    _caption_model = None
    _caption_load_err = traceback.format_exc()


# Emotion classifier
_emotion_pipe = None
try:
    _emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
except Exception:
    try:
        _emotion_pipe = pipeline("sentiment-analysis")
    except Exception:
        _emotion_pipe = None
        _emotion_load_err = traceback.format_exc()


# Generator for poetic lines (use small GPT-2 fallback)
_poem_pipe = None
try:
    _poem_pipe = pipeline("text-generation", model="gpt2")
except Exception:
    _poem_pipe = None
    _poem_load_err = traceback.format_exc()


# Object detection (yolov5 via torch.hub)
_obj_model = None
try:
    # this will download weights the first time — can be heavy
    _obj_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception:
    _obj_model = None
    _obj_load_err = traceback.format_exc()


# ---------- Public helper functions ----------

def caption_image(image: Image.Image) -> str:
    """
    Return a caption for a PIL image.
    If BLIP model isn't available, return a simple heuristic caption.
    """
    if _caption_processor is not None and _caption_model is not None:
        try:
            inputs = _caption_processor(images=image, return_tensors="pt")
            out = _caption_model.generate(**inputs, max_new_tokens=40)
            caption = _caption_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            # fallback below
            pass

    # fallback: simple placeholder based on size/colors
    try:
        w, h = image.size
        colors = image.convert("RGB").resize((30, 30)).getcolors(900)
        dominant = sorted(colors, key=lambda x: x[0], reverse=True)[0][1]
        return f"A {w}x{h} image with dominant color RGB{dominant}"
    except Exception:
        return "An image"


def detect_objects(frame) -> List[str]:
    """
    Accepts either a numpy array (H x W x C) or PIL image converted to numpy.
    Returns list of detected object names (may be empty).
    """
    if _obj_model is None:
        return []

    try:
        results = _obj_model(frame)
        names = results.pandas().xyxy[0]['name'].tolist()
        return names
    except Exception:
        return []


def analyze_emotion_text(text: str) -> str:
    """
    Return a one-word emotion label (lowercase) derived from text.
    Uses watsonx if client available; otherwise HF emotion model; otherwise 'neutral'.
    """
    # watsonx.ai path (optional)
    if client:
        try:
            prompt = f"Analyze the emotional tone of this text: {text}. Respond with one word (joy, anger, sadness, calm, love, fear, hope, neutral)."
            response = client.llms.generate(model_id="ibm/granite-13b-chat-v2", input=prompt, parameters={"decoding_method": "greedy", "max_new_tokens": 6})
            out = response['results'][0]['generated_text'].strip().lower()
            # sanitize output
            out = out.split()[0]
            return out
        except Exception:
            pass

    # HuggingFace pipeline
    if _emotion_pipe:
        try:
            res = _emotion_pipe(text)
            # pipeline returns list of dicts; label may be 'joy' or 'LABEL_0' depending on model
            label = res[0].get("label", "")
            return label.lower()
        except Exception:
            pass

    return "neutral"


def poetic_line(text: str) -> str:
    """
    Generate a short poetic line describing the provided text.
    Uses LLM pipeline if available, otherwise returns a template string.
    """
    if client:
        try:
            prompt = f"Write a short poetic line about: {text}"
            response = client.llms.generate(model_id="ibm/granite-13b-chat-v2", input=prompt, parameters={"decoding_method": "greedy", "max_new_tokens": 40})
            return response['results'][0]['generated_text'].strip()
        except Exception:
            pass

    if _poem_pipe:
        try:
            out = _poem_pipe(f"Write a poetic line about {text}", max_length=40, num_return_sequences=1)
            return out[0]["generated_text"].strip()
        except Exception:
            pass

    return f"A gentle thought about {text}."


def text_to_speech(text: str, output_path: str = "output.mp3") -> str:
    """
    Produce an MP3 using gTTS and return the path.
    Note: gTTS requires outbound access to Google TTS. If offline is required, swap for another engine.
    """
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(output_path)
        return output_path
    except Exception as e:
        # final fallback: write a tiny silent mp3 so UI can still play something
        try:
            with open(output_path, "wb") as f:
                f.write(b"")  # empty file
            return output_path
        except Exception:
            raise e


# Optional helper: map detected object -> instrument / sound label (simple mapping)
OBJECT_SOUND_MAP = {
    "person": "soft piano",
    "bicycle": "light bells",
    "car": "percussive drum",
    "dog": "wooden xylophone",
    "cat": "gentle flute",
    "bird": "bright piccolo",
    "cup": "mellow harp",
    "chair": "low bass",
    "laptop": "electronic pads",
    "book": "acoustic guitar",
    "phone": "synth beep",
}

def object_to_sound_label(object_name: Optional[str]) -> str:
    if not object_name:
        return "ambient pad"
    return OBJECT_SOUND_MAP.get(object_name.lower(), "ambient pad")
