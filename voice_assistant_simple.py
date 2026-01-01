# voice_assistant_simple.py - Super Reliable Version
import speech_recognition as sr
import subprocess
import threading
import queue
from typing import Optional
import time

class VoiceAssistant:
    """Simplified, reliable voice assistant"""
    
    def __init__(self, wake_word="hey assistant"):
        self.wake_word = wake_word.lower()
        self.recognizer = sr.Recognizer()
        
        # Better recognition settings
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
        # State
        self.is_listening = False
        self.command_queue = queue.Queue()
        self.microphone_lock = threading.Lock()
        
        # Callbacks
        self.on_command = None
        
        print("🎤 Microphone ready!")
    
    def speak(self, text: str):
        """Mac's built-in speech (most reliable)"""
        print(f"🔊 Assistant: {text}")
        try:
            # Use Mac's 'say' command - most reliable
            subprocess.run(['say', text], check=False)
        except Exception as e:
            print(f"Speech error: {e}")
    
    def listen_once(self, timeout: int = 5) -> Optional[str]:
        """Listen for single command"""
        if not self.microphone_lock.acquire(blocking=False):
            return None
        
        try:
            print("🎤 Listening... (speak clearly)")
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=5
                )
            
            print("🔄 Processing...")
            
            # Try multiple recognition services for better accuracy
            text = None
            
            # Try Google first
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"✅ Heard: {text}")
                return text.lower()
            except sr.UnknownValueError:
                print("❓ Didn't catch that. Try again?")
                return None
            except sr.RequestError as e:
                print(f"❌ Recognition error: {e}")
                return None
            
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
        finally:
            self.microphone_lock.release()
    
    def start_listening(self, callback=None):
        """Start background listening"""
        self.is_listening = True
        self.on_command = callback
        
        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()
        
        print(f"👂 Say '{self.wake_word}' to activate")
        self.speak(f"Ready. Say {self.wake_word}")
    
    def _listen_loop(self):
        """Background listening loop"""
        while self.is_listening:
            if not self.microphone_lock.acquire(blocking=False):
                time.sleep(0.5)
                continue
            
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=4)
                
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    
                    if self.wake_word in text:
                        print(f"🎤 Wake word detected!")
                        self.microphone_lock.release()
                        
                        self.speak("Yes?")
                        time.sleep(0.5)
                        
                        command = self.listen_once(timeout=5)
                        if command and self.on_command:
                            self.on_command(command)
                        
                        continue
                        
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    print(f"Recognition error: {e}")
                    
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                print(f"Loop error: {e}")
            finally:
                try:
                    self.microphone_lock.release()
                except:
                    pass
            
            time.sleep(0.1)
    
    def stop_listening(self):
        """Stop background listening"""
        self.is_listening = False
        self.speak("Deactivated")
    
    def process_command(self, command: str, emotion: str = "neutral") -> str:
        """Process command"""
        command = command.lower()
        
        if any(word in command for word in ['hello', 'hi', 'hey']):
            return "Hello! How are you?"
        
        elif any(word in command for word in ['recommend', 'suggest', 'show', 'find', 'play']):
            if 'funny' in command:
                return "Finding funny videos for you"
            elif 'calm' in command:
                return "Finding calming content"
            else:
                return f"Finding content for your {emotion} mood"
        
        elif 'help' in command:
            return "I can recommend videos based on your emotions"
        
        else:
            return "I'm here to help. What would you like?"


if __name__ == "__main__":
    print("Testing Voice Assistant\n")
    assistant = VoiceAssistant()
    
    assistant.speak("Testing speech")
    
    print("\nSay something:")
    text = assistant.listen_once(timeout=5)
    if text:
        print(f"Success: {text}")
        response = assistant.process_command(text)
        assistant.speak(response)