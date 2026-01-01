# advanced_features.py
from typing import List, Dict
import numpy as np
from datetime import datetime, timedelta

class UserProfile:
    """User ka emotion profile track karo"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.emotion_timeline = []
        self.watched_content = []
        self.preferences = {}
    
    def add_emotion(self, emotion: str, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()
        
        self.emotion_timeline.append({
            'emotion': emotion,
            'timestamp': timestamp
        })
    
    def get_emotion_pattern(self, hours: int = 24) -> Dict:
        """Last N hours ka emotion pattern"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_emotions = [
            e['emotion'] for e in self.emotion_timeline
            if e['timestamp'] > cutoff_time
        ]
        
        pattern = {}
        for emotion in recent_emotions:
            pattern[emotion] = pattern.get(emotion, 0) + 1
        
        return pattern
    
    def add_watched_content(self, content_id: str, rating: float = None):
        self.watched_content.append({
            'content_id': content_id,
            'timestamp': datetime.now(),
            'rating': rating
        })

class AdaptiveContentSelector:
    """Smart content selection based on user history"""
    
    def __init__(self):
        self.user_profiles = {}
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
        return self.user_profiles[user_id]
    
    def select_content(
        self, 
        user_id: str,
        current_emotion: str,
        available_content: List[Dict]
    ) -> List[Dict]:
        """User history aur current emotion se best content select karo"""
        
        profile = self.get_or_create_profile(user_id)
        profile.add_emotion(current_emotion)
        
        # Score each content
        scored_content = []
        for content in available_content:
            score = self._calculate_content_score(
                content,
                current_emotion,
                profile
            )
            scored_content.append({
                **content,
                'relevance_score': score
            })
        
        # Sort by score
        scored_content.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_content
    
    def _calculate_content_score(
        self,
        content: Dict,
        current_emotion: str,
        profile: UserProfile
    ) -> float:
        """Content ka relevance score calculate karo"""
        
        score = 0.0
        
        # Emotion match (40% weight)
        if current_emotion in content.get('emotions', []):
            score += 0.4
        
        # User history (30% weight)
        emotion_pattern = profile.get_emotion_pattern(hours=24)
        if emotion_pattern:
            common_emotions = set(content.get('emotions', [])) & set(emotion_pattern.keys())
            score += 0.3 * (len(common_emotions) / len(emotion_pattern))
        
        # Content freshness (20% weight)
        watched_ids = [w['content_id'] for w in profile.watched_content[-10:]]
        if content.get('url') not in watched_ids:
            score += 0.2
        
        # Category preference (10% weight)
        if content.get('category') in profile.preferences:
            score += 0.1 * profile.preferences[content['category']]
        
        return score