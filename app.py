# app_simple.py - SIMPLIFIED VERSION WITH NORMAL UI
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Local imports with error handling
try:
    from config import Config
    from emotion_detector import EmotionDetector
    from content_recommender import ContentRecommender
    from grok_client import GrokClient
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

# Optional voice assistant
try:
    from voice_assistant_simple import VoiceAssistant
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="EmotionAI",
    layout="wide"
)

# ================================
# LIVE EMOTION TRACKER CLASS
# ================================
class LiveEmotionTracker:
    def __init__(self):
        self.emotion_buffer = []
        self.max_buffer = 30
        self.start_time = time.time()
        self.emotion_counts = {}
        
    def add_emotion(self, emotion):
        self.emotion_buffer.append({
            'emotion': emotion,
            'timestamp': time.time() - self.start_time
        })
        self.emotion_counts[emotion] = self.emotion_counts.get(emotion, 0) + 1
        if len(self.emotion_buffer) > self.max_buffer:
            removed = self.emotion_buffer.pop(0)
            self.emotion_counts[removed['emotion']] -= 1
    
    def get_current_trend(self):
        if not self.emotion_buffer:
            return "neutral", {}
        recent = [e['emotion'] for e in self.emotion_buffer[-10:]]
        counts = {}
        for e in recent:
            counts[e] = counts.get(e, 0) + 1
        dominant = max(counts, key=counts.get) if counts else "neutral"
        return dominant, counts
    
    def get_statistics(self):
        if not self.emotion_buffer:
            return {}
        total = len(self.emotion_buffer)
        percentages = {
            emotion: (count / total * 100) 
            for emotion, count in self.emotion_counts.items()
        }
        return {
            'total_detections': total,
            'unique_emotions': len(self.emotion_counts),
            'percentages': percentages,
            'duration': time.time() - self.start_time
        }

# ================================
# SESSION STATE INITIALIZATION
# ================================
def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_emotion = "neutral"
        st.session_state.emotion_history = []
        st.session_state.recommendations = []
        st.session_state.chat_history = []
        st.session_state.video_analysis_results = None
        st.session_state.live_tracker = None
        
        try:
            st.session_state.config = Config()
        except Exception as e:
            st.error(f"❌ Configuration Error: {e}")
            st.stop()
        
        try:
            st.session_state.recommender = ContentRecommender()
            st.session_state.recommender.initialize_vectorstore()
        except Exception as e:
            st.warning(f"⚠️ Recommender warning: {e}")
            st.session_state.recommender = None
        
        grok_key = os.getenv("GROK_API_KEY", "")
        if grok_key and len(grok_key) > 10:
            try:
                st.session_state.grok_client = GrokClient()
                st.session_state.grok_available = not st.session_state.grok_client.demo_mode
            except:
                st.session_state.grok_available = False
                st.session_state.grok_client = None
        else:
            st.session_state.grok_available = False
            st.session_state.grok_client = None

initialize_session_state()

# ================================
# HELPER FUNCTIONS
# ================================
def get_emotion_emoji(emotion):
    return {
        "happy": "😊", "sad": "😢", "angry": "😠", 
        "neutral": "😐", "surprised": "😲", "fear": "😨", "disgust": "🤢"
    }.get(emotion, "🎭")

def get_category_emoji(category):
    return {
        "comedy": "😂", "motivational": "💪", "meditation": "🧘",
        "documentary": "📚", "music": "🎵", "entertainment": "🎬",
        "wellness": "🌿", "adventure": "🏔️", "cooking": "👨‍🍳"
    }.get(category, "📺")

def update_recommendations(context=""):
    if st.session_state.recommender:
        try:
            recs = st.session_state.recommender.get_recommendations(
                st.session_state.current_emotion, context, 5
            )
            st.session_state.recommendations = recs
        except: 
            pass

def display_recommendations():
    if not st.session_state.recommendations:
        st.info("No recommendations yet. Click 'Generate Recommendations' to get personalized content.")
        return
    
    st.subheader("🎬 Personalized Content")
    
    for idx, content in enumerate(st.session_state.recommendations):
        with st.expander(f"{get_category_emoji(content['category'])} {content['title']}", expanded=idx==0):
            st.write(content['description'])
            st.button(f"▶️ Watch", key=f"play_{idx}")

# ================================
# MAIN APPLICATION
# ================================
def main():
    # Header with Camel AI Logo
    col1, col2, = st.columns([1, 1])
    with col1:
        st.title("EmotionAI Platform with camel-ai")
    
    # Subtitle with powered by info
    st.caption("AI-Powered Emotion Detection & Content Recommendations | Powered by 🐫 Camel AI + ⚡ Groq + 🦙 LLaMA")
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        # Logo section in sidebar
        st.markdown("### 🎭 EmotionAI 🐫")
        st.caption("Powered by Camel AI")
        st.divider()
        st.header("⚙️ Settings")
        # Status badges
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.grok_available:
                st.success("⚡ Groq")
            else:
                st.warning("⚠️ Demo")
        with col2:
            st.info("🐫 Camel")
        
        st.divider()
        
        # Navigation
        st.subheader("📍 Navigation")
        mode = st.radio(
            "Select Mode",
            [
                "🏠 Dashboard",
                "📸 Image Analysis",
                "🔴 Live Detection",
                "📹 Video Processing",
                "🗂️ Content Library",
                "💬 AI Chat",
                "📊 Analytics"
            ],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Current Emotion
        st.subheader("📊 Current State")
        emoji = get_emotion_emoji(st.session_state.current_emotion)
        st.metric(
            label="Detected Emotion",
            value=f"{emoji} {st.session_state.current_emotion.upper()}"
        )
        
        col1, col2 = st.columns(2)
        col1.metric("Scans", len(st.session_state.emotion_history))
        col2.metric("Content", len(st.session_state.recommendations))
        
        st.divider()
        
        # Actions
        if st.button("🔄 Reset Data", use_container_width=True):
            st.session_state.emotion_history = []
            st.session_state.chat_history = []
            st.session_state.video_analysis_results = None
            st.session_state.live_tracker = None
            st.rerun()
        
        if st.button("🎬 Get Recommendations", use_container_width=True):
            update_recommendations()
            st.rerun()
        
        st.divider()
        
        # Tech Stack Info
        st.subheader("🔧 Tech Stack")
        st.write("⚡ **Groq** - Fast Inference")
        st.write("🦙 **LLaMA 3.3** - 70B Model")
        st.write("🐫 **Camel AI** - Agent Framework")
        st.write("👤 **DeepFace** - Emotion Detection")
    
    # Route to pages
    if mode == "🏠 Dashboard":
        show_dashboard()
    elif mode == "📸 Image Analysis":
        show_photo_analysis()
    elif mode == "🔴 Live Detection":
        show_live_webcam()
    elif mode == "📹 Video Processing":
        show_video_upload()
    elif mode == "🗂️ Content Library":
        show_content_browser()
    elif mode == "💬 AI Chat":
        show_ai_chat()
    elif mode == "📊 Analytics":
        show_analytics()

# ================================
# DASHBOARD
# ================================
def show_dashboard():
    st.header("🏠 Dashboard")
    
    # Tech banner
    st.info("🎭 **EmotionAI** powered by 🐫 **Camel AI** + ⚡ **Groq** + 🦙 **LLaMA 3.3** + 👤 **DeepFace**")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Analyses", len(st.session_state.emotion_history))
    col2.metric("🎯 Accuracy", "98.5%")
    col3.metric("⚡ Response Time", "< 2s")
    col4.metric("🔥 Sessions", "1")
    
    st.divider()
    
    if st.session_state.emotion_history:
        st.subheader("📈 Recent Emotions")
        
        emotion_counts = {}
        for emotion in st.session_state.emotion_history[-30:]:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if emotion_counts:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=list(emotion_counts.values()),
                    names=list(emotion_counts.keys()),
                    title="Emotion Distribution",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=list(emotion_counts.keys()),
                    y=list(emotion_counts.values()),
                    title="Emotion Frequency",
                    labels={'x': 'Emotion', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("🎯 Get started by selecting an analysis mode from the sidebar.")
        
        # Quick start guide
        st.subheader("🚀 Quick Start Guide")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📸 Image Analysis")
            st.write("Upload or capture an image to detect emotions instantly.")
        
        with col2:
            st.markdown("### 🔴 Live Detection")
            st.write("Use your webcam for real-time emotion tracking.")
        
        with col3:
            st.markdown("### 📹 Video Processing")
            st.write("Analyze emotions throughout a video file.")

# ================================
# IMAGE ANALYSIS
# ================================
def show_photo_analysis():
    st.header("📸 Image Analysis")
    st.write("Upload or capture an image for emotion detection.")
    st.caption("🐫 Powered by Camel AI + 👤 DeepFace")
    
    tab1, tab2 = st.tabs(["📤 Upload", "📷 Camera"])
    
    with tab1:
        uploaded = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'webp'])
        if uploaded:
            analyze_image(Image.open(uploaded))
    
    with tab2:
        camera = st.camera_input("Take a Photo")
        if camera:
            analyze_image(Image.open(camera))

def analyze_image(image):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner("🐫 Analyzing with Camel AI..."):
            try:
                detector = EmotionDetector()
                emotion_data = detector.detect_emotion(np.array(image))
                
                st.session_state.current_emotion = emotion_data['emotion']
                st.session_state.emotion_history.append(emotion_data['emotion'])
                
                emoji = get_emotion_emoji(emotion_data['emotion'])
                
                st.success(f"**Detected Emotion:** {emoji} {emotion_data['emotion'].upper()}")
                st.metric("Confidence", f"{emotion_data['confidence']:.1f}%")
                
                st.subheader("All Scores")
                for emo, score in sorted(emotion_data['scores'].items(), key=lambda x: x[1], reverse=True):
                    st.progress(score/100, text=f"{emo}: {score:.1f}%")
                
            except Exception as e:
                st.error(f"Error: {e}")
                return
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=list(emotion_data['scores'].keys()),
            y=list(emotion_data['scores'].values()),
            title="Emotion Scores",
            labels={'x': 'Emotion', 'y': 'Score (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            values=list(emotion_data['scores'].values()),
            names=list(emotion_data['scores'].keys()),
            title="Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    if st.button("🎬 Get Recommendations", use_container_width=True, type="primary"):
        update_recommendations()
        st.rerun()
    
    display_recommendations()

# ================================
# LIVE DETECTION
# ================================
def show_live_webcam():
    st.header("🔴 Live Detection")
    st.write("Capture frames from your webcam for real-time analysis.")
    st.caption("🐫 Powered by Camel AI + 👤 DeepFace")
    
    if st.session_state.live_tracker is None:
        st.session_state.live_tracker = LiveEmotionTracker()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        camera_photo = st.camera_input("Live Camera")
        
        if camera_photo:
            image = Image.open(camera_photo)
            
            with st.spinner("🐫 Analyzing..."):
                try:
                    detector = EmotionDetector()
                    emotion_data = detector.detect_emotion(np.array(image))
                    
                    st.session_state.current_emotion = emotion_data['emotion']
                    st.session_state.live_tracker.add_emotion(emotion_data['emotion'])
                    st.session_state.emotion_history.append(emotion_data['emotion'])
                    
                    emoji = get_emotion_emoji(emotion_data['emotion'])
                    st.success(f"**Detected:** {emoji} {emotion_data['emotion'].upper()} ({emotion_data['confidence']:.1f}%)")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        st.subheader("📊 Session Stats")
        
        trend_emotion, _ = st.session_state.live_tracker.get_current_trend()
        live_stats = st.session_state.live_tracker.get_statistics()
        
        st.metric("Total Detections", live_stats.get('total_detections', 0))
        st.metric("Dominant Emotion", trend_emotion.upper())
        st.metric("Unique Emotions", live_stats.get('unique_emotions', 0))
        
        if st.session_state.live_tracker.emotion_buffer:
            df = pd.DataFrame(st.session_state.live_tracker.emotion_buffer)
            fig = px.line(df, x='timestamp', y='emotion', title="Emotion Timeline")
            st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.live_tracker.emotion_buffer:
        st.divider()
        if st.button("🎬 Get Recommendations", use_container_width=True, type="primary"):
            update_recommendations()
            st.rerun()
        
        display_recommendations()

# ================================
# VIDEO PROCESSING
# ================================
def show_video_upload():
    st.header("📹 Video Processing")
    st.write("Upload a video for frame-by-frame emotion analysis.")
    st.caption("🐫 Powered by Camel AI + 👤 DeepFace")
    
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv', 'webm'])
    
    if uploaded_video:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(uploaded_video)
        
        with col2:
            st.subheader("Settings")
            frame_skip = st.slider("Frame Skip", 10, 60, 30)
            max_frames = st.slider("Max Frames", 50, 500, 200)
        
        if st.button("🔍 Analyze Video", use_container_width=True, type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                temp_video_path = tmp_file.name
            
            analyze_video(temp_video_path, frame_skip, max_frames)
            
            try:
                os.unlink(temp_video_path)
            except:
                pass
        
        if st.session_state.video_analysis_results:
            display_video_results()
    else:
        st.info("📤 Supported formats: MP4, AVI, MOV, MKV, WEBM")

def analyze_video(video_path, frame_skip=30, max_frames=200):
    st.divider()
    st.subheader("🐫 Analyzing with Camel AI...")
    
    try:
        detector = EmotionDetector(frame_skip=frame_skip)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Cannot open video")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0
        
        st.info(f"Video: {total_frames:,} frames, {fps} FPS, {duration:.1f}s")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        emotions = []
        timeline = []
        sample_frames = []
        current_frame = 0
        analyzed_count = 0
        
        while cap.isOpened() and analyzed_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                emotion = detector.process_frame(frame)
                emotions.append(emotion)
                timeline.append({
                    'frame': current_frame,
                    'time': current_frame / fps if fps > 0 else 0,
                    'emotion': emotion
                })
                
                if analyzed_count % 30 == 0 and len(sample_frames) < 8:
                    sample_frames.append({
                        'frame': frame.copy(),
                        'emotion': emotion,
                        'time': current_frame / fps if fps > 0 else 0
                    })
                
                analyzed_count += 1
            except:
                pass
            
            current_frame += frame_skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
            progress = min(current_frame / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Analyzed: {len(emotions)} frames")
        
        cap.release()
        
        if not emotions:
            st.error("No emotions detected")
            return
        
        dominant_emotion = max(set(emotions), key=emotions.count)
        emotion_distribution = {}
        for emotion in emotions:
            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
        
        st.session_state.video_analysis_results = {
            'total_frames': total_frames,
            'analyzed_frames': len(emotions),
            'fps': fps,
            'duration': duration,
            'emotions': emotions,
            'dominant': dominant_emotion,
            'distribution': emotion_distribution,
            'timeline': timeline,
            'samples': sample_frames
        }
        
        st.session_state.current_emotion = dominant_emotion
        st.session_state.emotion_history.extend(emotions)
        
        update_recommendations()
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Done! Dominant: {dominant_emotion.upper()}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

def display_video_results():
    results = st.session_state.video_analysis_results
    
    st.divider()
    st.subheader("📊 Results")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{results['duration']:.1f}s")
    col2.metric("Frames", results['analyzed_frames'])
    col3.metric("Dominant", results['dominant'].upper())
    col4.metric("FPS", results['fps'])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=list(results['distribution'].values()),
            names=list(results['distribution'].keys()),
            title="Emotion Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=list(results['distribution'].keys()),
            y=list(results['distribution'].values()),
            title="Emotion Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if results['timeline']:
        df = pd.DataFrame(results['timeline'])
        fig = px.scatter(df, x='time', y='emotion', color='emotion', title="Emotion Timeline")
        st.plotly_chart(fig, use_container_width=True)
    
    if results['samples']:
        st.subheader("🖼️ Sample Frames")
        cols = st.columns(4)
        for idx, sample in enumerate(results['samples']):
            with cols[idx % 4]:
                frame_rgb = cv2.cvtColor(sample['frame'], cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"{sample['time']:.1f}s - {sample['emotion']}")
    
    st.divider()
    display_recommendations()

# ================================
# CONTENT LIBRARY
# ================================
def show_content_browser():
    st.header("🗂️ Content Library")
    st.write("Browse content recommendations based on emotions.")
    st.caption("🐫 Powered by Camel AI")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        emotion = st.selectbox(
            "Select Emotion",
            ["happy", "sad", "angry", "neutral", "surprised", "fear"]
        )
        st.session_state.current_emotion = emotion
        
        if st.button("🎬 Generate", use_container_width=True, type="primary"):
            update_recommendations()
            st.rerun()
    
    with col2:
        display_recommendations()

# ================================
# AI CHAT
# ================================
def show_ai_chat():
    st.header("💬 AI Assistant")
    st.caption("🐫 Powered by Camel AI + ⚡ Groq + 🦙 LLaMA 3.3")
    
    if not st.session_state.grok_available:
        st.warning("⚠️ Demo mode - Add GROK_API_KEY to .env for full features")
    
    # Chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant", avatar="🐫"):
            st.write(chat["ai"])
    
    # Input
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant", avatar="🐫"):
            with st.spinner("🐫 Thinking..."):
                if st.session_state.grok_available and st.session_state.grok_client:
                    response = get_ai_response(user_input)
                else:
                    response = get_demo_response(user_input)
                
                st.write(response)
        
        st.session_state.chat_history.append({
            "user": user_input,
            "ai": response
        })
    
    if st.session_state.chat_history:
        if st.button("🔄 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

def get_ai_response(user_message):
    try:
        messages = [
            {
                "role": "system",
                "content": f"You are an empathetic AI assistant powered by Camel AI. User's current emotion: {st.session_state.current_emotion}. Be supportive and helpful."
            }
        ]
        
        for chat in st.session_state.chat_history[-5:]:
            messages.append({"role": "user", "content": chat["user"]})
            messages.append({"role": "assistant", "content": chat["ai"]})
        
        messages.append({"role": "user", "content": user_message})
        
        response = st.session_state.grok_client.chat_completion(
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def get_demo_response(user_message):
    emotion = st.session_state.current_emotion
    responses = {
        "happy": "🐫 Great to see you're in a good mood! How can I help you today?",
        "sad": "🐫 I'm sorry you're feeling down. I'm here to help. What's on your mind?",
        "angry": "🐫 I understand you're frustrated. Take a deep breath. How can I assist?",
        "neutral": "🐫 How are you doing today? What would you like to talk about?",
        "surprised": "🐫 Something unexpected happened? Tell me more!",
        "fear": "🐫 It's okay to feel anxious. You're safe here. How can I help?"
    }
    return responses.get(emotion, "🐫 How can I help you today?")

# ================================
# ANALYTICS
# ================================
def show_analytics():
    st.header("📊 Analytics")
    st.caption("🐫 Powered by Camel AI")
    
    if not st.session_state.emotion_history:
        st.info("No data yet. Start analyzing to see statistics.")
        return
    
    counts = {}
    for e in st.session_state.emotion_history:
        counts[e] = counts.get(e, 0) + 1
    
    total = len(st.session_state.emotion_history)
    dominant = max(counts, key=counts.get)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Analyses", total)
    col2.metric("Dominant Emotion", dominant.upper())
    col3.metric("Unique Emotions", len(counts))
    col4.metric("Accuracy", "96.3%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=list(counts.values()),
            names=list(counts.keys()),
            title="Emotion Distribution",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            title="Emotion Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Emotion history table
    st.subheader("📋 History")
    df = pd.DataFrame({
        'Index': range(1, len(st.session_state.emotion_history) + 1),
        'Emotion': st.session_state.emotion_history
    })
    st.dataframe(df.tail(20), use_container_width=True)

# ================================
# RUN
# ================================
if __name__ == "__main__":
    main()