# config.py
import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    VIDEOS_DIR = DATA_DIR / "videos"
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"
    
    # Grok API Configuration
    GROK_API_KEY = os.getenv("GROK_API_KEY", "your-grok-api-key-here")
    GROK_API_BASE = "https://api.x.ai/v1"  # Grok API endpoint
    GROK_MODEL = "grok-beta"  # Grok model name
    
    # Alternative: Anthropic Claude (agar backup chahiye)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Local Embeddings (FREE - No API needed)
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # FREE local model
    USE_LOCAL_EMBEDDINGS = True
    
    # Emotion Detection
    EMOTION_DETECTION_MODEL = "DeepFace"
    DETECTION_BACKEND = "opencv"
    FRAME_SKIP = 30
    
    # Video Settings
    SUPPORTED_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
    MAX_VIDEO_SIZE = 500 * 1024 * 1024
    
    # RAG Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # MCP Server
    MCP_HOST = "localhost"
    MCP_PORT = 8000
    
    # Content Categories by Emotion
    EMOTION_CONTENT_MAP = {
        "happy": ["comedy", "entertainment", "music", "adventure"],
        "sad": ["motivational", "uplifting", "comedy", "inspiring"],
        "angry": ["calm", "meditation", "nature", "relaxing"],
        "neutral": ["educational", "documentary", "news"],
        "surprised": ["trending", "viral", "amazing"],
        "fear": ["comforting", "positive", "heartwarming"]
    }
    
    # Grok-specific prompts
    SYSTEM_PROMPTS = {
        "recommendation": """You are an AI assistant for an emotion-adaptive video streaming platform. 
        Based on the user's current emotion, suggest appropriate video content that would help improve 
        their mood or maintain their positive state. Be empathetic and personalized.""",
        
        "content_analysis": """Analyze the given video content and determine which emotions 
        it would be most suitable for. Consider the content's tone, theme, and potential emotional impact."""
    }
    
    # Create directories
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)