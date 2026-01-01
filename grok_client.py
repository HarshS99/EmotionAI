# grok_client.py - COMPLETE PRODUCTION VERSION
"""
Groq API Client for EmotionAI Platform
Handles AI chat completions using Groq's Llama 3.3 70B model
"""

from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GrokClient:
    """
    Groq API client for AI-powered conversations
    Supports both API mode and demo mode for testing
    """
    
    def __init__(self):
        """Initialize Groq client with API key validation"""
        self.api_key = os.getenv("GROK_API_KEY", "")
        
        print("🔍 Checking API key...")
        print(f"   Length: {len(self.api_key)} characters")
        
        if not self.api_key or len(self.api_key) < 10:
            print("⚠️ Groq API key not found or invalid")
            print("   Running in DEMO MODE")
            self.demo_mode = True
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                print("✅ Groq API connected successfully!")
                self.demo_mode = False
            except Exception as e:
                print(f"❌ Groq initialization error: {e}")
                print("   Falling back to DEMO MODE")
                self.demo_mode = True
                self.client = None
    
    def chat_completion(self, messages, max_tokens=500, temperature=0.1, model="llama-3.3-70b-versatile"):
        """
        Get chat completion from Groq API
        
        Args:
            messages (list): List of message dicts with 'role' and 'content'
                Example: [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello!"}
                ]
            max_tokens (int): Maximum tokens in response (default: 1000)
            temperature (float): Response randomness 0-1 (default: 0.7)
            model (str): Groq model to use (default: "llama-3.3-70b-versatile")
            
        Returns:
            str: AI response text
        """
        if self.demo_mode or not self.client:
            return self._demo_response(messages)
        
        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stream=False
            )
            
            # Extract response text
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                return "I apologize, but I couldn't generate a proper response. Please try again."
                
        except Exception as e:
            print(f"❌ Groq API Error: {str(e)}")
            return f"I encountered an error while processing your request. Please try again. (Error: {str(e)})"
    
    def _demo_response(self, messages):
        """
        Demo response when API is not available
        Provides basic pattern-matching responses for testing
        
        Args:
            messages (list): List of message dicts
            
        Returns:
            str: Demo response text
        """
        if not messages:
            return "Hello! How can I help you today?"
        
        # Get last user message
        last_message = messages[-1]["content"].lower()
        
        # Simple pattern matching for demo
        if "hello" in last_message or "hi" in last_message or "hey" in last_message:
            return "Hello! I'm here to help you. How are you feeling today?"
        
        elif "help" in last_message or "support" in last_message:
            return "I'm here to provide emotional support and guidance. Tell me what's on your mind, and I'll do my best to help."
        
        elif "recommend" in last_message or "suggest" in last_message:
            return "Based on your emotional state, I'd recommend taking some time for self-care activities. Would you like specific suggestions?"
        
        elif "happy" in last_message or "good" in last_message or "great" in last_message:
            return "That's wonderful to hear! It's great that you're feeling positive. Would you like suggestions to maintain this good mood?"
        
        elif "sad" in last_message or "down" in last_message or "depressed" in last_message:
            return "I'm sorry you're feeling this way. It's okay to feel sad sometimes. Would you like to talk about it or get some suggestions for uplifting activities?"
        
        elif "angry" in last_message or "mad" in last_message or "frustrated" in last_message:
            return "I understand you're feeling frustrated. Anger is a valid emotion. Would you like some techniques to help calm down?"
        
        elif "thank" in last_message or "thanks" in last_message:
            return "You're very welcome! I'm always here if you need more support. Is there anything else I can help you with?"
        
        elif "bye" in last_message or "goodbye" in last_message:
            return "Take care! Remember, I'm here whenever you need support. Have a wonderful day!"
        
        else:
            return "I understand. I'm here to listen and support you. Could you tell me more about what you're experiencing or what you'd like help with?"
    
    def get_emotion_support(self, emotion, context=""):
        """
        Get emotion-specific support message
        
        Args:
            emotion (str): Current emotion (e.g., "happy", "sad", "angry")
            context (str): Optional context about the situation
            
        Returns:
            str: Supportive AI response
        """
        prompt = f"The user is currently feeling {emotion}. "
        if context:
            prompt += f"Context: {context}. "
        prompt += "Provide empathetic support, understanding, and practical suggestions to help them."
        
        messages = [
            {"role": "system", "content": "You are an empathetic emotional support AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        return self.chat_completion(messages, max_tokens=500)
    
    def analyze_conversation(self, conversation_history):
        """
        Analyze conversation for emotional insights
        
        Args:
            conversation_history (list): List of chat message dicts
                Each dict should have 'user' and 'ai' keys
                
        Returns:
            str: Analysis and insights
        """
        if not conversation_history:
            return "No conversation history available."
        
        # Build summary from last 5 exchanges
        conv_text = "\n".join([
            f"User: {msg['user']}\nAI: {msg['ai']}" 
            for msg in conversation_history[-5:]
        ])
        
        prompt = f"""Analyze this conversation and provide insights:

{conv_text}

Provide:
1. Main emotional themes
2. User's primary concerns
3. Suggestions for continued support"""
        
        messages = [
            {"role": "system", "content": "You are an expert in emotional analysis and psychology."},
            {"role": "user", "content": prompt}
        ]
        
        return self.chat_completion(messages, max_tokens=600)
    
    def get_mood_recommendations(self, emotion, preferences=None):
        """
        Get activity recommendations based on mood
        
        Args:
            emotion (str): Current emotion
            preferences (list): Optional user preferences
            
        Returns:
            str: Personalized recommendations
        """
        prompt = f"The user is feeling {emotion}. "
        if preferences:
            prompt += f"Their preferences include: {', '.join(preferences)}. "
        prompt += "Suggest 5 specific activities or content that would be beneficial for this emotional state."
        
        messages = [
            {"role": "system", "content": "You are a helpful AI that provides personalized activity recommendations."},
            {"role": "user", "content": prompt}
        ]
        
        return self.chat_completion(messages, max_tokens=400)
    
    def generate_affirmation(self, emotion):
        """
        Generate positive affirmation based on emotion
        
        Args:
            emotion (str): Current emotion
            
        Returns:
            str: Positive affirmation message
        """
        prompt = f"Generate a short, powerful positive affirmation for someone feeling {emotion}. Keep it under 50 words and make it uplifting."
        
        messages = [
            {"role": "system", "content": "You are an expert at creating motivational affirmations."},
            {"role": "user", "content": prompt}
        ]
        
        return self.chat_completion(messages, max_tokens=100, temperature=0.8)
    
    def explain_emotion(self, emotion, context=""):
        """
        Explain an emotion and its causes
        
        Args:
            emotion (str): Emotion to explain
            context (str): Optional context
            
        Returns:
            str: Explanation of the emotion
        """
        prompt = f"Explain the emotion '{emotion}' in simple terms. "
        if context:
            prompt += f"Context: {context}. "
        prompt += "Include what causes it, how it feels, and why it's a valid emotion. Keep it concise and supportive."
        
        messages = [
            {"role": "system", "content": "You are a psychology expert who explains emotions clearly."},
            {"role": "user", "content": prompt}
        ]
        
        return self.chat_completion(messages, max_tokens=500)
    
    def get_coping_strategies(self, emotion):
        """
        Get coping strategies for specific emotion
        
        Args:
            emotion (str): Current emotion
            
        Returns:
            str: Coping strategies and techniques
        """
        prompt = f"Provide 5 practical coping strategies for someone feeling {emotion}. Make them actionable and easy to implement."
        
        messages = [
            {"role": "system", "content": "You are a mental health expert providing coping strategies."},
            {"role": "user", "content": prompt}
        ]
        
        return self.chat_completion(messages, max_tokens=600)
    
    def is_available(self):
        """
        Check if API is available
        
        Returns:
            bool: True if API is available, False if in demo mode
        """
        return not self.demo_mode
    
    def get_status(self):
        """
        Get current client status
        
        Returns:
            dict: Status information
        """
        return {
            "demo_mode": self.demo_mode,
            "api_available": not self.demo_mode,
            "model": "llama-3.3-70b-versatile" if not self.demo_mode else "demo",
            "status": "Connected" if not self.demo_mode else "Demo Mode"
        }


# ================================
# EXAMPLE USAGE
# ================================
if __name__ == "__main__":
    """
    Example usage of GrokClient
    """
    print("=" * 50)
    print("Testing Groq Client")
    print("=" * 50)
    
    # Initialize client
    client = GrokClient()
    
    # Check status
    status = client.get_status()
    print(f"\n📊 Status: {status}")
    
    # Test chat completion
    print("\n💬 Testing Chat Completion:")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you?"}
    ]
    response = client.chat_completion(messages)
    print(f"Response: {response}")
    
    # Test emotion support
    print("\n🎭 Testing Emotion Support:")
    support = client.get_emotion_support("sad", "Had a rough day at work")
    print(f"Support: {support}")
    
    # Test affirmation
    print("\n✨ Testing Affirmation:")
    affirmation = client.generate_affirmation("happy")
    print(f"Affirmation: {affirmation}")
    
    # Test coping strategies
    print("\n🧘 Testing Coping Strategies:")
    strategies = client.get_coping_strategies("anxious")
    print(f"Strategies: {strategies}")
    
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print("=" * 50)