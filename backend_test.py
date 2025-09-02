#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for Bujji AI Vision Assistant
Tests all endpoints, AI integrations, and error handling
"""

import requests
import sys
import json
import base64
import io
import time
from datetime import datetime
from PIL import Image
import tempfile
import os

class BujjiAPITester:
    def __init__(self, base_url="https://ai-sight-buddy.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.client_id = f"test_client_{int(time.time())}"
        self.tests_run = 0
        self.tests_passed = 0
        self.scene_analyzed = False
        
    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED {details}")
        else:
            print(f"❌ {name} - FAILED {details}")
        return success

    def create_test_image(self):
        """Create a simple test image as base64"""
        try:
            # Create a simple test image
            img = Image.new('RGB', (640, 480), color='blue')
            
            # Add some text/shapes to make it more interesting
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Draw some shapes
            draw.rectangle([50, 50, 200, 150], fill='red', outline='white', width=3)
            draw.ellipse([300, 100, 500, 300], fill='green', outline='white', width=3)
            
            # Add text
            try:
                # Try to use default font
                draw.text((100, 200), "TEST IMAGE", fill='white')
            except:
                # If font fails, just draw without text
                pass
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80)
            img_data = buffer.getvalue()
            
            return base64.b64encode(img_data).decode()
            
        except Exception as e:
            print(f"Error creating test image: {e}")
            return None

    def create_test_audio(self):
        """Create a simple test audio file"""
        try:
            # Create a simple WAV file with silence
            import wave
            import struct
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            
            # WAV file parameters
            sample_rate = 16000
            duration = 2  # 2 seconds
            frequency = 440  # A4 note
            
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(sample_rate)
                
                # Generate a simple tone
                for i in range(int(sample_rate * duration)):
                    value = int(32767 * 0.1)  # Low volume
                    wav_file.writeframes(struct.pack('<h', value))
            
            return temp_file.name
            
        except Exception as e:
            print(f"Error creating test audio: {e}")
            return None

    def test_health_check(self):
        """Test the health check endpoint"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['status', 'whisper_loaded', 'vision_loaded', 'timestamp']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return self.log_test("Health Check", False, f"Missing fields: {missing_fields}")
                
                # Check if AI models are loaded
                whisper_ok = data.get('whisper_loaded', False)
                vision_ok = data.get('vision_loaded', False)
                
                details = f"Status: {data['status']}, Whisper: {whisper_ok}, Vision: {vision_ok}"
                
                if data['status'] == 'healthy' and whisper_ok and vision_ok:
                    return self.log_test("Health Check", True, details)
                else:
                    return self.log_test("Health Check", False, f"Unhealthy - {details}")
            else:
                return self.log_test("Health Check", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            return self.log_test("Health Check", False, f"Exception: {e}")

    def test_scene_analysis(self):
        """Test scene analysis with a test image"""
        try:
            # Create test image
            test_image_b64 = self.create_test_image()
            if not test_image_b64:
                return self.log_test("Scene Analysis", False, "Could not create test image")
            
            # Prepare request
            payload = {
                "image_data": test_image_b64,
                "client_id": self.client_id
            }
            
            response = requests.post(
                f"{self.api_url}/analyze-scene",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30  # AI analysis can take time
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['description', 'timestamp', 'confidence']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return self.log_test("Scene Analysis", False, f"Missing fields: {missing_fields}")
                
                description = data.get('description', '')
                confidence = data.get('confidence', 0)
                
                if description and len(description) > 10:  # Reasonable description length
                    self.scene_analyzed = True
                    return self.log_test("Scene Analysis", True, f"Description: '{description[:50]}...', Confidence: {confidence}")
                else:
                    return self.log_test("Scene Analysis", False, f"Invalid description: '{description}'")
            else:
                error_text = response.text if response.text else "No error details"
                return self.log_test("Scene Analysis", False, f"HTTP {response.status_code}: {error_text}")
                
        except Exception as e:
            return self.log_test("Scene Analysis", False, f"Exception: {e}")

    def test_question_answering(self):
        """Test Q&A functionality (requires prior scene analysis)"""
        try:
            if not self.scene_analyzed:
                return self.log_test("Question Answering", False, "No scene analyzed yet - skipping Q&A test")
            
            # Test question
            test_question = "What colors do you see in the image?"
            
            payload = {
                "question": test_question,
                "client_id": self.client_id
            }
            
            response = requests.post(
                f"{self.api_url}/ask-question",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30  # AI processing can take time
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['answer', 'scene_context', 'timestamp']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    return self.log_test("Question Answering", False, f"Missing fields: {missing_fields}")
                
                answer = data.get('answer', '')
                scene_context = data.get('scene_context', '')
                
                if answer and len(answer) > 5 and scene_context:
                    return self.log_test("Question Answering", True, f"Answer: '{answer[:50]}...', Context available: {bool(scene_context)}")
                else:
                    return self.log_test("Question Answering", False, f"Invalid response - Answer: '{answer}', Context: {bool(scene_context)}")
            else:
                error_text = response.text if response.text else "No error details"
                return self.log_test("Question Answering", False, f"HTTP {response.status_code}: {error_text}")
                
        except Exception as e:
            return self.log_test("Question Answering", False, f"Exception: {e}")

    def test_audio_transcription(self):
        """Test audio transcription with a test audio file"""
        try:
            # Create test audio file
            audio_file_path = self.create_test_audio()
            if not audio_file_path:
                return self.log_test("Audio Transcription", False, "Could not create test audio file")
            
            try:
                # Upload audio file
                with open(audio_file_path, 'rb') as audio_file:
                    files = {'file': ('test_audio.wav', audio_file, 'audio/wav')}
                    
                    response = requests.post(
                        f"{self.api_url}/transcribe-audio",
                        files=files,
                        timeout=30  # Whisper processing can take time
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check required fields
                    required_fields = ['transcription', 'language']
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        return self.log_test("Audio Transcription", False, f"Missing fields: {missing_fields}")
                    
                    transcription = data.get('transcription', '')
                    language = data.get('language', '')
                    
                    # For a silent/tone audio, transcription might be empty or minimal
                    return self.log_test("Audio Transcription", True, f"Transcription: '{transcription}', Language: {language}")
                else:
                    error_text = response.text if response.text else "No error details"
                    return self.log_test("Audio Transcription", False, f"HTTP {response.status_code}: {error_text}")
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(audio_file_path)
                except:
                    pass
                    
        except Exception as e:
            return self.log_test("Audio Transcription", False, f"Exception: {e}")

    def test_error_handling(self):
        """Test error handling with invalid requests"""
        tests_passed = 0
        total_error_tests = 0
        
        # Test 1: Invalid image data for scene analysis
        total_error_tests += 1
        try:
            payload = {"image_data": "invalid_base64", "client_id": self.client_id}
            response = requests.post(f"{self.api_url}/analyze-scene", json=payload, timeout=10)
            
            if response.status_code in [400, 422]:  # Expected error codes
                tests_passed += 1
                print(f"✅ Error Handling (Invalid Image) - PASSED (HTTP {response.status_code})")
            else:
                print(f"❌ Error Handling (Invalid Image) - FAILED (Expected 400/422, got {response.status_code})")
        except Exception as e:
            print(f"❌ Error Handling (Invalid Image) - FAILED (Exception: {e})")
        
        # Test 2: Q&A without scene context
        total_error_tests += 1
        try:
            payload = {"question": "What do you see?", "client_id": "nonexistent_client"}
            response = requests.post(f"{self.api_url}/ask-question", json=payload, timeout=10)
            
            if response.status_code in [400, 404]:  # Expected error codes
                tests_passed += 1
                print(f"✅ Error Handling (No Scene Context) - PASSED (HTTP {response.status_code})")
            else:
                print(f"❌ Error Handling (No Scene Context) - FAILED (Expected 400/404, got {response.status_code})")
        except Exception as e:
            print(f"❌ Error Handling (No Scene Context) - FAILED (Exception: {e})")
        
        # Test 3: Invalid audio file
        total_error_tests += 1
        try:
            files = {'file': ('test.txt', b'not an audio file', 'text/plain')}
            response = requests.post(f"{self.api_url}/transcribe-audio", files=files, timeout=10)
            
            if response.status_code in [400, 422]:  # Expected error codes
                tests_passed += 1
                print(f"✅ Error Handling (Invalid Audio) - PASSED (HTTP {response.status_code})")
            else:
                print(f"❌ Error Handling (Invalid Audio) - FAILED (Expected 400/422, got {response.status_code})")
        except Exception as e:
            print(f"❌ Error Handling (Invalid Audio) - FAILED (Exception: {e})")
        
        # Update overall test counts
        self.tests_run += total_error_tests
        self.tests_passed += tests_passed
        
        return tests_passed == total_error_tests

    def test_websocket_connection(self):
        """Test WebSocket connection (basic connectivity test)"""
        try:
            import websocket
            
            ws_url = f"{self.base_url.replace('https://', 'wss://').replace('http://', 'ws://')}/ws/{self.client_id}"
            
            # Simple connection test
            ws = websocket.create_connection(ws_url, timeout=10)
            
            # Send ping message
            ping_msg = json.dumps({"type": "ping"})
            ws.send(ping_msg)
            
            # Wait for response
            response = ws.recv()
            data = json.loads(response)
            
            ws.close()
            
            if data.get("type") == "pong":
                return self.log_test("WebSocket Connection", True, "Ping-pong successful")
            else:
                return self.log_test("WebSocket Connection", False, f"Unexpected response: {data}")
                
        except ImportError:
            return self.log_test("WebSocket Connection", False, "websocket-client not available - skipping")
        except Exception as e:
            return self.log_test("WebSocket Connection", False, f"Exception: {e}")

    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting Bujji AI Vision Assistant Backend Tests")
        print(f"📡 Testing API at: {self.api_url}")
        print(f"🆔 Client ID: {self.client_id}")
        print("=" * 60)
        
        # Core API tests
        self.test_health_check()
        self.test_scene_analysis()
        self.test_question_answering()
        self.test_audio_transcription()
        
        # Error handling tests
        print("\n🔍 Testing Error Handling...")
        self.test_error_handling()
        
        # WebSocket test
        print("\n🔌 Testing WebSocket...")
        self.test_websocket_connection()
        
        # Final results
        print("\n" + "=" * 60)
        print(f"📊 FINAL RESULTS: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 ALL TESTS PASSED! Backend is working correctly.")
            return 0
        else:
            failed_tests = self.tests_run - self.tests_passed
            print(f"⚠️  {failed_tests} test(s) failed. Please check the issues above.")
            return 1

def main():
    """Main test runner"""
    tester = BujjiAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())