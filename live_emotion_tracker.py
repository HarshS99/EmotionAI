# live_emotion_tracker.py - Real-time continuous emotion tracking
import cv2
import numpy as np
from deepface import DeepFace
import threading
import queue
from collections import deque
import time
from typing import Dict, Optional, Callable

class LiveEmotionTracker:
    """Continuous real-time emotion tracking"""
    
    def __init__(self, camera_index=0, fps=10):
        self.camera_index = camera_index
        self.fps = fps
        self.frame_delay = 1.0 / fps
        
        # Video capture
        self.cap = None
        self.is_running = False
        
        # Current state
        self.current_frame = None
        self.current_emotion = "neutral"
        self.emotion_confidence = 0.0
        self.emotion_scores = {}
        
        # History
        self.emotion_history = deque(maxlen=100)
        self.emotion_timeline = deque(maxlen=1000)
        
        # Callbacks
        self.on_emotion_change = None
        self.on_frame_processed = None
        
        # Thread safety
        self.lock = threading.Lock()
        
    def start(self):
        """Start live tracking"""
        if self.is_running:
            print("⚠️ Already running")
            return
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise Exception("❌ Cannot open camera")
        
        self.is_running = True
        
        # Start capture thread
        capture_thread = threading.Thread(target=self._capture_loop)
        capture_thread.daemon = True
        capture_thread.start()
        
        # Start processing thread
        process_thread = threading.Thread(target=self._process_loop)
        process_thread.daemon = True
        process_thread.start()
        
        print("✅ Live emotion tracking started")
    
    def _capture_loop(self):
        """Continuous frame capture"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame
            time.sleep(self.frame_delay)
    
    def _process_loop(self):
        """Continuous emotion processing"""
        while self.is_running:
            frame = None
            with self.lock:
                if self.current_frame is not None:
                    frame = self.current_frame.copy()
            
            if frame is not None:
                try:
                    # Detect emotion
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
                    confidence = scores[emotion]
                    
                    # Update state
                    previous_emotion = self.current_emotion
                    
                    with self.lock:
                        self.current_emotion = emotion
                        self.emotion_confidence = confidence
                        self.emotion_scores = scores
                        self.emotion_history.append(emotion)
                        self.emotion_timeline.append({
                            'timestamp': time.time(),
                            'emotion': emotion,
                            'confidence': confidence
                        })
                    
                    # Callback on emotion change
                    if emotion != previous_emotion and self.on_emotion_change:
                        self.on_emotion_change(emotion, confidence)
                    
                    # Frame processed callback
                    if self.on_frame_processed:
                        self.on_frame_processed(frame, emotion, scores)
                    
                except Exception as e:
                    print(f"Detection error: {e}")
            
            time.sleep(1.0)  # Process every second
    
    def get_current_state(self) -> Dict:
        """Get current emotion state"""
        with self.lock:
            return {
                'emotion': self.current_emotion,
                'confidence': self.emotion_confidence,
                'scores': self.emotion_scores.copy(),
                'history': list(self.emotion_history)
            }
    
    def get_dominant_emotion(self, window_size: int = 10) -> str:
        """Get most common recent emotion"""
        with self.lock:
            recent = list(self.emotion_history)[-window_size:]
            if not recent:
                return "neutral"
            
            counts = {}
            for emotion in recent:
                counts[emotion] = counts.get(emotion, 0) + 1
            
            return max(counts, key=counts.get)
    
    def get_emotion_distribution(self) -> Dict[str, float]:
        """Get emotion distribution percentages"""
        with self.lock:
            if not self.emotion_history:
                return {}
            
            total = len(self.emotion_history)
            counts = {}
            
            for emotion in self.emotion_history:
                counts[emotion] = counts.get(emotion, 0) + 1
            
            return {
                emotion: (count / total) * 100
                for emotion, count in counts.items()
            }
    
    def stop(self):
        """Stop tracking"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        print("🛑 Live tracking stopped")
    
    def get_frame_with_overlay(self) -> Optional[np.ndarray]:
        """Get current frame with emotion overlay"""
        with self.lock:
            if self.current_frame is None:
                return None
            
            frame = self.current_frame.copy()
            
            # Add emotion text
            emotion_text = f"{self.current_emotion.upper()} ({self.emotion_confidence:.1f}%)"
            cv2.putText(
                frame,
                emotion_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Add emotion bars
            y_offset = 70
            for emotion, score in sorted(self.emotion_scores.items(), key=lambda x: x[1], reverse=True):
                bar_width = int(score * 3)
                cv2.rectangle(
                    frame,
                    (10, y_offset),
                    (10 + bar_width, y_offset + 20),
                    (0, 255, 0) if emotion == self.current_emotion else (100, 100, 100),
                    -1
                )
                cv2.putText(
                    frame,
                    f"{emotion}: {score:.0f}%",
                    (10, y_offset + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                y_offset += 30
            
            return frame