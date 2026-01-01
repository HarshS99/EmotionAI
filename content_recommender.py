# content_recommender.py - Simplified Version WITHOUT huggingface downloads
from typing import List, Dict
import numpy as np
from config import Config
import json

# Simple local embeddings using basic NLP
class SimpleLocalEmbeddings:
    """Simple keyword-based embeddings - NO external dependencies"""
    
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenization"""
        return text.lower().split()
    
    def _build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        all_words = set()
        for text in texts:
            words = self._tokenize(text)
            all_words.update(words)
        
        self.vocab = {word: idx for idx, word in enumerate(all_words)}
        self.vocab_size = len(self.vocab)
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to simple bag-of-words vector"""
        vector = np.zeros(self.vocab_size)
        words = self._tokenize(text)
        
        for word in words:
            if word in self.vocab:
                vector[self.vocab[word]] += 1
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        self._build_vocab(texts)
        return [self._text_to_vector(text).tolist() for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._text_to_vector(text).tolist()
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class ContentRecommender:
    """Smart content recommender WITHOUT external model downloads"""
    
    def __init__(self):
        self.config = Config()
        
        print("📥 Initializing local embeddings...")
        self.embeddings = SimpleLocalEmbeddings()
        print("✅ Embeddings ready!")
        
        # Try to load Grok client
        try:
            from grok_client import GrokClient
            self.grok_client = GrokClient()
            print("✅ Grok client initialized!")
        except Exception as e:
            print(f"⚠️ Grok client not available: {e}")
            self.grok_client = None
        
        self.content_database = self._load_content_database()
        self.content_embeddings = {}
        
    def _load_content_database(self) -> List[Dict]:
        """Load enhanced content database"""
        contents = [
            {
                "id": 1,
                "title": "Motivational Success Stories",
                "category": "motivational",
                "description": "Inspiring stories of people who overcame challenges and achieved their dreams through hard work and determination",
                "tags": ["inspiring", "uplifting", "success", "achievement", "motivation"],
                "duration": "10:30",
                "url": "motivational_1.mp4",
                "emotions": ["sad", "neutral"],
                "intensity": "high",
                "mood_boost": 8
            },
            {
                "id": 2,
                "title": "Stand-up Comedy Special",
                "category": "comedy",
                "description": "Hilarious stand-up comedy to brighten your day and make you laugh out loud",
                "tags": ["funny", "entertaining", "laughter", "humor", "jokes"],
                "duration": "25:00",
                "url": "comedy_1.mp4",
                "emotions": ["sad", "angry", "happy"],
                "intensity": "high",
                "mood_boost": 9
            },
            {
                "id": 3,
                "title": "Peaceful Nature Meditation",
                "category": "meditation",
                "description": "Calm nature scenes with soothing music for deep relaxation and inner peace",
                "tags": ["relaxing", "peaceful", "nature", "calm", "zen", "meditation"],
                "duration": "15:00",
                "url": "meditation_1.mp4",
                "emotions": ["angry", "fear", "neutral"],
                "intensity": "low",
                "mood_boost": 7
            },
            {
                "id": 4,
                "title": "Technology Innovation Documentary",
                "category": "documentary",
                "description": "Fascinating insights into cutting-edge technology and future innovations",
                "tags": ["educational", "informative", "tech", "science", "innovation"],
                "duration": "30:00",
                "url": "tech_doc_1.mp4",
                "emotions": ["neutral", "surprised"],
                "intensity": "medium",
                "mood_boost": 6
            },
            {
                "id": 5,
                "title": "Live Music Festival Highlights",
                "category": "music",
                "description": "Energetic performances from top music festivals around the world",
                "tags": ["energetic", "music", "entertainment", "live", "concert"],
                "duration": "20:00",
                "url": "music_fest_1.mp4",
                "emotions": ["happy", "neutral"],
                "intensity": "high",
                "mood_boost": 8
            },
            {
                "id": 6,
                "title": "Cute Animals Compilation",
                "category": "entertainment",
                "description": "Adorable and funny moments with cute animals that will warm your heart",
                "tags": ["cute", "animals", "funny", "heartwarming", "pets"],
                "duration": "12:00",
                "url": "animals_1.mp4",
                "emotions": ["sad", "angry", "neutral"],
                "intensity": "medium",
                "mood_boost": 9
            },
            {
                "id": 7,
                "title": "Yoga and Breathing Exercises",
                "category": "wellness",
                "description": "Guided yoga and breathing exercises for stress relief and mental clarity",
                "tags": ["yoga", "wellness", "breathing", "stress-relief", "health"],
                "duration": "18:00",
                "url": "yoga_1.mp4",
                "emotions": ["angry", "fear", "sad"],
                "intensity": "low",
                "mood_boost": 7
            },
            {
                "id": 8,
                "title": "Adventure Travel Vlog",
                "category": "adventure",
                "description": "Exciting travel adventures to exotic destinations around the globe",
                "tags": ["travel", "adventure", "exciting", "exploration", "world"],
                "duration": "22:00",
                "url": "travel_1.mp4",
                "emotions": ["neutral", "happy", "surprised"],
                "intensity": "high",
                "mood_boost": 8
            },
            {
                "id": 9,
                "title": "Quick Cooking Recipes",
                "category": "cooking",
                "description": "Easy and delicious recipes you can make in under 30 minutes",
                "tags": ["cooking", "food", "recipes", "quick", "delicious"],
                "duration": "15:00",
                "url": "cooking_1.mp4",
                "emotions": ["neutral", "happy"],
                "intensity": "medium",
                "mood_boost": 6
            },
            {
                "id": 10,
                "title": "Inspirational TED Talks",
                "category": "motivational",
                "description": "Life-changing ideas and perspectives from world-renowned speakers",
                "tags": ["inspiring", "educational", "ideas", "wisdom", "talks"],
                "duration": "35:00",
                "url": "ted_talks_1.mp4",
                "emotions": ["sad", "neutral", "surprised"],
                "intensity": "medium",
                "mood_boost": 8
            },
            {
                "id": 11,
                "title": "Relaxing Ocean Waves",
                "category": "meditation",
                "description": "Peaceful ocean sounds and visuals for ultimate relaxation",
                "tags": ["ocean", "relaxing", "nature", "peaceful", "calm"],
                "duration": "60:00",
                "url": "ocean_1.mp4",
                "emotions": ["angry", "fear", "neutral"],
                "intensity": "low",
                "mood_boost": 8
            },
            {
                "id": 12,
                "title": "Action Movie Highlights",
                "category": "entertainment",
                "description": "Thrilling action sequences from blockbuster movies",
                "tags": ["action", "exciting", "thrilling", "movies", "entertainment"],
                "duration": "20:00",
                "url": "action_1.mp4",
                "emotions": ["neutral", "happy", "surprised"],
                "intensity": "high",
                "mood_boost": 7
            }
        ]
        
        return contents
    
    def initialize_vectorstore(self):
        """Initialize simple vector store"""
        print("🔧 Building content embeddings...")
        
        # Create embeddings for all content
        all_texts = []
        for content in self.content_database:
            text = f"{content['title']} {content['description']} {' '.join(content['tags'])}"
            all_texts.append(text)
        
        # Build vocabulary and embeddings
        embeddings = self.embeddings.embed_documents(all_texts)
        
        # Store embeddings with content IDs
        for idx, content in enumerate(self.content_database):
            self.content_embeddings[content['id']] = embeddings[idx]
        
        print("✅ Content embeddings ready!")
    
    def get_recommendations(
        self, 
        emotion: str, 
        context: str = "",
        num_results: int = 5
    ) -> List[Dict]:
        """Get smart recommendations based on emotion"""
        
        # Step 1: Filter by emotion match
        preferred_categories = self.config.EMOTION_CONTENT_MAP.get(
            emotion, 
            ["general"]
        )
        
        suitable_content = [
            content for content in self.content_database
            if emotion in content['emotions'] or 
               content['category'] in preferred_categories
        ]
        
        # Step 2: If context provided, use semantic similarity
        if context and self.content_embeddings:
            query_embedding = self.embeddings.embed_query(
                f"{emotion} {context}"
            )
            
            # Calculate similarities
            scored_content = []
            for content in suitable_content:
                content_embedding = self.content_embeddings.get(content['id'], [])
                
                if content_embedding:
                    similarity = self.embeddings.similarity(
                        query_embedding, 
                        content_embedding
                    )
                    scored_content.append({
                        **content,
                        'similarity_score': similarity
                    })
                else:
                    scored_content.append({
                        **content,
                        'similarity_score': 0.5
                    })
            
            # Sort by similarity and mood boost
            scored_content.sort(
                key=lambda x: (x['similarity_score'], x['mood_boost']),
                reverse=True
            )
            
            suitable_content = scored_content
        else:
            # Step 3: Sort by mood boost
            suitable_content.sort(
                key=lambda x: x['mood_boost'],
                reverse=True
            )
        
        results = suitable_content[:num_results]
        
        # Step 4: Add Grok insights if available
        if self.grok_client and results:
            try:
                grok_insight = self.grok_client.get_recommendation(
                    emotion=emotion,
                    context=context,
                    available_content=results[:3]
                )
                
                if results:
                    results[0]['ai_insight'] = grok_insight
                    
            except Exception as e:
                print(f"⚠️ Grok enhancement skipped: {e}")
        
        return results
    
    def get_personalized_message(self, emotion: str) -> str:
        """Get personalized message for emotion"""
        
        # Try Grok first
        if self.grok_client:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an empathetic AI. Provide a brief, supportive message (1-2 sentences)."
                    },
                    {
                        "role": "user",
                        "content": f"Short empathetic message for someone feeling {emotion}."
                    }
                ]
                
                return self.grok_client.chat_completion(messages, max_tokens=100)
                
            except Exception as e:
                print(f"⚠️ Grok message error: {e}")
        
        # Fallback messages
        messages = {
            "happy": "Great to see you happy! Here are some entertaining videos to keep the vibe going! 😊",
            "sad": "We understand you're feeling down. These uplifting videos might help brighten your day. 💙",
            "angry": "Take a deep breath. These calming videos can help you relax. 🧘‍♂️",
            "neutral": "Here are some interesting videos you might enjoy! 📺",
            "surprised": "You seem amazed! Check out these fascinating videos! 😲",
            "fear": "Everything will be okay. Here are some comforting videos. 🤗"
        }
        
        return messages.get(emotion, "Here are some recommended videos for you!")


# Test function
if __name__ == "__main__":
    print("🧪 Testing Content Recommender...")
    
    recommender = ContentRecommender()
    recommender.initialize_vectorstore()
    
    print("\n📝 Testing recommendations for 'sad' emotion...")
    recommendations = recommender.get_recommendations("sad", "User had a tough day at work")
    
    print(f"\n✅ Found {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"  {rec['id']}. {rec['title']} (Mood Boost: {rec['mood_boost']}/10)")
    
    print("\n✅ Test completed successfully!")