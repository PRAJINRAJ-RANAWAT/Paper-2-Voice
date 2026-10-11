"""
Configuration module for Paper to Voice Assistant
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Google API Key setup
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception:
        pass

# Model configuration
GOOGLE_MODEL_NAME = os.getenv('GOOGLE_MODEL_NAME', 'gemini-1.5-flash')

# Retry and concurrency configuration
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
BASE_RETRY_DELAY = float(os.getenv('BASE_RETRY_DELAY', '1.0'))
MAX_RETRY_DELAY = float(os.getenv('MAX_RETRY_DELAY', '30.0'))
MAX_CONCURRENCY = int(os.getenv('MAX_CONCURRENCY', '4'))

# Distributed Celery configuration
USE_CELERY = os.getenv('USE_CELERY', 'false').lower() in ('true', '1', 'yes')
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
CELERY_TASK_TIMEOUT = int(os.getenv('CELERY_TASK_TIMEOUT', '120'))

# Audio Configuration
AUDIO_SAMPLE_RATE = int(os.getenv('AUDIO_SAMPLE_RATE', '44100'))
AUDIO_CHANNELS = int(os.getenv('AUDIO_CHANNELS', '2'))  # 2 for stereo
LIGHT_GUITAR_FREQ = 440
AMBIENT_GUITAR_FREQ = 220
GUITAR_DURATION = 1000  # milliseconds

# TTS Configuration
TTS_MODEL = os.getenv('TTS_MODEL', "myshell-ai/MeloTTS-English")

# File paths
TEMP_DIR = os.getenv('TEMP_DIR', "temp")
VOICES_DIR = os.getenv('VOICES_DIR', "voices")


def get_llm(model_name: str | None = None, max_retries: int = 1) -> ChatGoogleGenerativeAI:
    """
    Get configured Gemini Chat model instance.
    
    Args:
        model_name: Optional override for model name.
        max_retries: SDK-level retries. Set low to avoid compounding with application-level retries.
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    return ChatGoogleGenerativeAI(
        model=model_name or GOOGLE_MODEL_NAME,
        temperature=0,
        max_tokens=None,
        max_retries=max_retries,
    )
