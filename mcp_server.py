# mcp_server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
from datetime import datetime
from emotion_detector import EmotionDetector
from content_recommender import ContentRecommender
from grok_client import GrokClient

app = FastAPI(
    title="Emotion-Adaptive Streaming MCP Server (Grok-Powered)",
    description="Real-time emotion detection and AI-powered content recommendations"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
print("🚀 Initializing MCP Server...")
emotion_detector = EmotionDetector()
content_recommender = ContentRecommender()
content_recommender.initialize_vectorstore()

try:
    grok_client = GrokClient()
    print("✅ Grok AI connected!")
except Exception as e:
    print(f"⚠️ Grok not available: {e}")
    grok_client = None

# Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Models
class EmotionRequest(BaseModel):
    emotion: str
    confidence: float
    context: Optional[str] = ""
    user_id: Optional[str] = "default"

class ChatRequest(BaseModel):
    message: str
    emotion: Optional[str] = "neutral"
    context: Optional[str] = ""

# Routes
@app.get("/")
async def root():
    return {
        "message": "Emotion-Adaptive Streaming MCP Server (Grok-Powered)",
        "version": "2.0.0",
        "ai_engine": "Grok + Local Embeddings",
        "status": "running",
        "grok_available": grok_client is not None
    }

@app.post("/analyze-emotion")
async def analyze_emotion(request: EmotionRequest):
    """Emotion analyze aur Grok-powered recommendations"""
    
    # Get recommendations
    recommendations = content_recommender.get_recommendations(
        emotion=request.emotion,
        context=request.context,
        num_results=5
    )
    
    # Personalized message
    message = content_recommender.get_personalized_message(request.emotion)
    
    # Grok insights (agar available ho)
    grok_insights = None
    if grok_client:
        try:
            grok_insights = grok_client.get_recommendation(
                emotion=request.emotion,
                context=request.context,
                available_content=recommendations
            )
        except Exception as e:
            print(f"⚠️ Grok insights error: {e}")
    
    return {
        "emotion": request.emotion,
        "confidence": request.confidence,
        "message": message,
        "recommendations": recommendations,
        "grok_insights": grok_insights,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    """Grok ke saath chat karo"""
    
    if not grok_client:
        raise HTTPException(status_code=503, detail="Grok AI not available")
    
    try:
        messages = [
            {
                "role": "system",
                "content": f"You are an empathetic AI assistant for a video streaming platform. The user is feeling {request.emotion}. Be supportive and helpful."
            },
            {
                "role": "user",
                "content": request.message
            }
        ]
        
        response = grok_client.chat_completion(messages)
        
        return {
            "response": response,
            "emotion": request.emotion,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/content/{emotion}")
async def get_content_by_emotion(emotion: str, limit: int = 10):
    """Specific emotion ke liye content"""
    
    recommendations = content_recommender.get_recommendations(
        emotion=emotion,
        num_results=limit
    )
    
    return {
        "emotion": emotion,
        "count": len(recommendations),
        "content": recommendations
    }

@app.websocket("/ws/emotions")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time emotion updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "emotion_update":
                emotion = data.get("emotion")
                context = data.get("context", "")
                
                # Recommendations
                recommendations = content_recommender.get_recommendations(
                    emotion=emotion,
                    context=context
                )
                
                # Grok insights
                grok_insights = None
                if grok_client:
                    try:
                        grok_insights = grok_client.get_recommendation(
                            emotion=emotion,
                            context=context,
                            available_content=recommendations[:3]
                        )
                    except:
                        pass
                
                await websocket.send_json({
                    "type": "recommendations",
                    "emotion": emotion,
                    "recommendations": recommendations,
                    "grok_insights": grok_insights,
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "grok_available": grok_client is not None,
        "embeddings": "local (sentence-transformers)",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting MCP Server with Grok AI...")
    print("📍 Server: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)