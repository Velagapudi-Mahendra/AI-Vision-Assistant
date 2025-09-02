from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import logging
import json
import base64
import asyncio
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import io
import whisper
import cv2

# Import emergent integrations
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="Bujji AI Vision Assistant", version="1.0.0")

# Create API router
api_router = APIRouter(prefix="/api")

# Global variables for AI models
whisper_model = None
vision_chat = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_sessions: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_sessions[client_id] = {
            "last_scene_description": "",
            "conversation_history": []
        }
        logging.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_sessions:
            del self.client_sessions[client_id]
        logging.info(f"Client {client_id} disconnected")

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logging.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()

# Pydantic models
class SceneAnalysisRequest(BaseModel):
    image_data: str  # Base64 encoded image
    client_id: str

class SceneAnalysisResponse(BaseModel):
    description: str
    timestamp: datetime
    confidence: float

class QuestionRequest(BaseModel):
    question: str
    client_id: str

class QuestionResponse(BaseModel):
    answer: str
    scene_context: str
    timestamp: datetime

# Initialize AI models
@app.on_event("startup")
async def startup_event():
    global whisper_model, vision_chat
    try:
        # Load Whisper model
        logging.info("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        logging.info("Whisper model loaded successfully")
        
        # Initialize vision chat
        logging.info("Initializing vision AI...")
        emergent_key = os.environ.get('EMERGENT_LLM_KEY')
        if not emergent_key:
            raise ValueError("EMERGENT_LLM_KEY not found in environment")
        
        vision_chat = LlmChat(
            api_key=emergent_key,
            session_id="vision_assistant",
            system_message="You are Bujji, an AI vision assistant. You analyze images from a webcam and describe what you see in natural, conversational language. Focus on people, objects, activities, and the overall scene. Be concise but descriptive."
        ).with_model("openai", "gpt-4o")
        
        logging.info("Vision AI initialized successfully")
        
    except Exception as e:
        logging.error(f"Startup error: {e}")
        raise

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Bujji AI Vision Assistant API"}

@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper_loaded": whisper_model is not None,
        "vision_loaded": vision_chat is not None,
        "timestamp": datetime.utcnow()
    }

@api_router.post("/analyze-scene", response_model=SceneAnalysisResponse)
async def analyze_scene(request: SceneAnalysisRequest):
    """Analyze a scene from camera image"""
    try:
        if not vision_chat:
            raise HTTPException(status_code=503, detail="Vision AI not available")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image_data.split(',')[1] if ',' in request.image_data else request.image_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
        
        # Create image content for vision AI
        image_base64 = base64.b64encode(image_data).decode()
        image_content = ImageContent(image_base64=image_base64)
        
        # Create new chat instance for this request
        scene_chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=f"scene_{request.client_id}_{datetime.utcnow().timestamp()}",
            system_message="You are Bujji, an AI vision assistant. Describe what you see in this webcam image in 1-2 natural sentences. Focus on the most important elements - people, objects, activities, or scenes. Be conversational and engaging."
        ).with_model("openai", "gpt-4o")
        
        # Analyze the scene
        user_message = UserMessage(
            text="What do you see in this image? Describe it naturally in 1-2 sentences.",
            file_contents=[image_content]
        )
        
        response = await scene_chat.send_message(user_message)
        description = response.strip()
        
        # Store the scene description for this client
        if request.client_id in manager.client_sessions:
            manager.client_sessions[request.client_id]["last_scene_description"] = description
        
        return SceneAnalysisResponse(
            description=description,
            timestamp=datetime.utcnow(),
            confidence=0.85  # Placeholder confidence score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Scene analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Scene analysis failed: {e}")

@api_router.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio using Whisper"""
    try:
        if not whisper_model:
            raise HTTPException(status_code=503, detail="Whisper model not available")
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio format")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe with Whisper
            result = whisper_model.transcribe(
                temp_file_path,
                fp16=False,
                language="en"
            )
            
            return {
                "transcription": result["text"].strip(),
                "language": result.get("language", "en"),
                "segments": result.get("segments", [])
            }
            
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Audio transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

@api_router.post("/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Answer a question about the current scene"""
    try:
        if not vision_chat:
            raise HTTPException(status_code=503, detail="Vision AI not available")
        
        # Get the last scene description for this client
        scene_context = ""
        if request.client_id in manager.client_sessions:
            scene_context = manager.client_sessions[request.client_id]["last_scene_description"]
        
        if not scene_context:
            raise HTTPException(status_code=400, detail="No recent scene analysis available. Please capture a scene first.")
        
        # Create a Q&A chat instance
        qa_chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id=f"qa_{request.client_id}_{datetime.utcnow().timestamp()}",
            system_message=f"You are Bujji, an AI vision assistant. You have just analyzed a scene and described it as: '{scene_context}'. Now answer the user's question about this scene. Be conversational and helpful. If the question can't be answered from the scene description, say so politely."
        ).with_model("openai", "gpt-4o")
        
        # Answer the question
        user_message = UserMessage(
            text=f"Based on what you described about the scene ('{scene_context}'), please answer this question: {request.question}"
        )
        
        response = await qa_chat.send_message(user_message)
        answer = response.strip()
        
        # Store in conversation history
        if request.client_id in manager.client_sessions:
            manager.client_sessions[request.client_id]["conversation_history"].append({
                "question": request.question,
                "answer": answer,
                "scene_context": scene_context,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return QuestionResponse(
            answer=answer,
            scene_context=scene_context,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Question answering error: {e}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {e}")

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive and handle any messages
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "ping":
                await manager.send_message(client_id, {"type": "pong"})
            elif data.get("type") == "scene_update":
                # Handle real-time scene updates if needed
                await manager.send_message(client_id, {
                    "type": "scene_acknowledged",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logging.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)

# Include router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()