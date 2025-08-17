# ibm_client.py
import os
import time
import requests
from ibm_watson import NaturalLanguageUnderstandingV1, TextToSpeechV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

IAM_TOKEN_CACHE = {"token": None, "expires_at": 0}

def get_iam_token_from_apikey(apikey: str):
    """
    Exchanges an IBM Cloud API key for an IAM access token (server-side).
    Returns the token string and expiration epoch.
    """
    if IAM_TOKEN_CACHE["token"] and IAM_TOKEN_CACHE["expires_at"] - 30 > time.time():
        return IAM_TOKEN_CACHE["token"]
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": apikey
    }
    resp = requests.post(url, data=data, headers=headers, timeout=10)
    resp.raise_for_status()
    j = resp.json()
    token = j["access_token"]
    expires_in = j.get("expiration", None)  # some responses include an expiration epoch
    # store approximate expiry (1 hour typical)
    IAM_TOKEN_CACHE["token"] = token
    IAM_TOKEN_CACHE["expires_at"] = time.time() + 3500
    return token

def nlu_analyze_text(apikey, url, text):
    """
    Uses Watson NLU to get emotion scores.
    """
    authenticator = IAMAuthenticator(apikey)
    nlu = NaturalLanguageUnderstandingV1(
        version='2021-08-01',
        authenticator=authenticator
    )
    nlu.set_service_url(url)
    response = nlu.analyze(text=text, features=Features(emotion=EmotionOptions())).get_result()
    # Response structure => response["emotion"]["document"]["emotion"] etc.
    return response

def tts_synthesize(apikey, url, text, accept='audio/wav', voice='en-US_AllisonV3Voice'):
    """
    Returns bytes of synthesized audio using IBM TTS (server-side).
    """
    authenticator = IAMAuthenticator(apikey)
    tts = TextToSpeechV1(authenticator=authenticator)
    tts.set_service_url(url)
    response = tts.synthesize(text, voice=voice, accept=accept).get_result()
    audio_bytes = response.content
    return audio_bytes
