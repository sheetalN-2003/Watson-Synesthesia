import cv2
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from TTS.api import TTS
from ibm_watsonx_ai import APIClient
import os
from PIL import Image

# ---------- Watsonx.ai Setup ----------
IBM_APIKEY = os.getenv("IBM_APIKEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"

credentials = {"url": WATSONX_URL, "apikey": IBM_APIKEY}
client = APIClient(credentials)
client.set.default_project(PROJECT_ID)

# ---------- HuggingFace Models ----------
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# ---------- TTS Setup ----------
tts = TTS("tts_models/en/vctk/vits")

# ---------- Object Detection ----------
obj_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def analyze_emotion_text(text: str):
    """Get emotion from text using watsonx.ai or fallback HuggingFace"""
    try:
        prompt = f"Analyze the emotional tone of this text: {text}. Respond with one word (joy, anger, sadness, calm, love, fear, hope)."
        response = client.llms.generate(
            model_id="ibm/granite-13b-chat-v2",
            input=prompt,
            parameters={"decoding_method": "greedy", "max_new_tokens": 5}
        )
        return response['results'][0]['generated_text'].strip().lower()
    except:
        return emotion_analyzer(text)[0]["label"].lower()

def poetic_line(text: str):
    """Generate poetic description"""
    prompt = f"Write a short poetic line about: {text}"
    try:
        response = client.llms.generate(
            model_id="ibm/granite-13b-chat-v2",
            input=prompt,
            parameters={"decoding_method": "greedy", "max_new_tokens": 30}
        )
        return response['results'][0]['generated_text'].strip()
    except:
        return f"A poetic thought about {text}."

def caption_image(image: Image.Image):
    """Generate caption for uploaded image"""
    inputs = caption_processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)

def detect_objects(frame):
    """Detect objects in live camera frame"""
    results = obj_model(frame)
    return results.pandas().xyxy[0]['name'].tolist()

def text_to_speech(text: str, output="output.wav"):
    tts.tts_to_file(text=text, file_path=output)
    return output
