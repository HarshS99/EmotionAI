# emotion_detector.py - Simplified version
import cv2
import numpy as np
from deepface import DeepFace
from collections import deque
from typing import Dict
import threading

class EmotionDetector:
    def __init__(self, frame_skip=30):
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.current_emotion = "neutral"
        self.emotion_history = deque(maxlen=10)
        self.emotion_scores = {}
        self.lock = threading.Lock()
        
    def detect_emotion(self, frame) -> Dict:
        """Detect emotion from frame"""
        try:
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotion = result['dominant_emotion']
            scores = result['emotion']
            
            return {
                'emotion': emotion,
                'scores': scores,
                'confidence': scores[emotion]
            }
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return {
                'emotion': 'neutral',
                'scores': {},
                'confidence': 0
            }
    
    def process_frame(self, frame):
        """Process frame and detect emotion"""
        self.frame_count += 1
        
        if self.frame_count % self.frame_skip != 0:
            return self.current_emotion
        
        emotion_data = self.detect_emotion(frame)
        
        with self.lock:
            self.current_emotion = emotion_data['emotion']
            self.emotion_scores = emotion_data['scores']
            self.emotion_history.append(emotion_data['emotion'])
        
        return self.current_emotion
    
    def get_dominant_emotion(self) -> str:
        """Get most common recent emotion"""
        with self.lock:
            if not self.emotion_history:
                return "neutral"
            
            emotion_counts = {}
            for emotion in self.emotion_history:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            return max(emotion_counts, key=emotion_counts.get)

class WebcamEmotionDetector:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.detector = EmotionDetector()
    
    def get_current_emotion(self):
        return self.detector.get_dominant_emotion()