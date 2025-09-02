import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Mic, MicOff, Volume2, VolumeX, Eye, MessageSquare, Loader2, AlertTriangle } from 'lucide-react';
import { Button } from './components/ui/button';
import { Card } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { useToast } from './hooks/use-toast';
import { Toaster } from './components/ui/toaster';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function BujjiVisionAssistant() {
  // State management
  const [isScanning, setIsScanning] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [currentDescription, setCurrentDescription] = useState('');
  const [lastQuestion, setLastQuestion] = useState('');
  const [lastAnswer, setLastAnswer] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isProcessingQuestion, setIsProcessingQuestion] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [clientId] = useState(() => `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const speechRecognitionRef = useRef(null);
  const synthRef = useRef(null);
  const analysisIntervalRef = useRef(null);

  const { toast } = useToast();

  // Initialize speech synthesis
  useEffect(() => {
    if ('speechSynthesis' in window) {
      synthRef.current = window.speechSynthesis;
    }
  }, []);

  // Initialize camera
  const initializeCamera = useCallback(async () => {
    try {
      setCameraError('');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false
      });
      
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      toast({
        title: "Camera ready",
        description: "Webcam initialized successfully",
      });
      
    } catch (error) {
      console.error('Camera initialization error:', error);
      setCameraError('Camera access denied or not available');
      toast({
        variant: "destructive",
        title: "Camera Error",
        description: "Unable to access camera. Please check permissions.",
      });
    }
  }, [toast]);

  // Initialize speech recognition
  const initializeSpeechRecognition = useCallback(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      speechRecognitionRef.current = new SpeechRecognition();
      
      speechRecognitionRef.current.continuous = false;
      speechRecognitionRef.current.interimResults = false;
      speechRecognitionRef.current.lang = 'en-US';
      
      speechRecognitionRef.current.onresult = async (event) => {
        const transcript = event.results[0][0].transcript;
        setLastQuestion(transcript);
        setIsListening(false);
        
        toast({
          title: "Question received",
          description: transcript,
        });
        
        await processQuestion(transcript);
      };
      
      speechRecognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        toast({
          variant: "destructive",
          title: "Speech Recognition Error",
          description: "Could not process speech. Please try again.",
        });
      };
      
      speechRecognitionRef.current.onend = () => {
        setIsListening(false);
      };
    }
  }, [toast]);

  // Capture frame from video
  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return null;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
  }, []);

  // Analyze scene
  const analyzeScene = useCallback(async () => {
    if (isAnalyzing) return;
    
    const imageData = captureFrame();
    if (!imageData) return;
    
    setIsAnalyzing(true);
    
    try {
      const response = await fetch(`${API}/analyze-scene`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_data: imageData,
          client_id: clientId
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status}`);
      }
      
      const data = await response.json();
      setCurrentDescription(data.description);
      
      // Speak the description
      if (synthRef.current && !isSpeaking) {
        speakText(data.description);
      }
      
    } catch (error) {
      console.error('Scene analysis error:', error);
      toast({
        variant: "destructive",
        title: "Analysis Error",
        description: "Failed to analyze scene. Please try again.",
      });
    } finally {
      setIsAnalyzing(false);
    }
  }, [captureFrame, clientId, isSpeaking, toast]);

  // Process question
  const processQuestion = useCallback(async (question) => {
    if (isProcessingQuestion) return;
    
    setIsProcessingQuestion(true);
    setLastAnswer('');
    
    try {
      const response = await fetch(`${API}/ask-question`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          client_id: clientId
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Question processing failed: ${response.status}`);
      }
      
      const data = await response.json();
      setLastAnswer(data.answer);
      
      // Speak the answer
      if (synthRef.current && !isSpeaking) {
        speakText(data.answer);
      }
      
    } catch (error) {
      console.error('Question processing error:', error);
      const errorMessage = "I'm sorry, I couldn't process your question. Please make sure I've analyzed a recent scene and try again.";
      setLastAnswer(errorMessage);
      toast({
        variant: "destructive",
        title: "Question Error",  
        description: errorMessage,
      });
    } finally {
      setIsProcessingQuestion(false);
    }
  }, [clientId, isSpeaking, toast]);

  // Text-to-speech
  const speakText = useCallback((text) => {
    if (!synthRef.current || isSpeaking) return;
    
    // Cancel any ongoing speech
    synthRef.current.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    utterance.volume = 0.8;
    
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    
    synthRef.current.speak(utterance);
  }, [isSpeaking]);

  // Start/stop scanning
  const toggleScanning = useCallback(() => {
    if (isScanning) {
      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current);
        analysisIntervalRef.current = null;
      }
      setIsScanning(false);
      
      toast({
        title: "Scanning stopped",
        description: "Continuous scene analysis disabled",
      });
    } else {
      // Start continuous analysis
      analysisIntervalRef.current = setInterval(analyzeScene, 5000); // Every 5 seconds
      setIsScanning(true);
      
      toast({
        title: "Scanning started",
        description: "I'm now continuously analyzing what I see",
      });
      
      // Do immediate analysis
      analyzeScene();
    }
  }, [isScanning, analyzeScene, toast]);

  // Start/stop listening
  const toggleListening = useCallback(() => {
    if (!speechRecognitionRef.current) {
      toast({
        variant: "destructive",
        title: "Speech Recognition Unavailable",
        description: "Your browser doesn't support speech recognition",
      });
      return;
    }
    
    if (isListening) {
      speechRecognitionRef.current.stop();
      setIsListening(false);
    } else {
      speechRecognitionRef.current.start();
      setIsListening(true);
      
      toast({
        title: "Listening...",
        description: "Ask me a question about what I see",
      });
    }
  }, [isListening, toast]);

  // Stop speaking
  const stopSpeaking = useCallback(() => {
    if (synthRef.current) {
      synthRef.current.cancel();
      setIsSpeaking(false);
    }
  }, []);

  // Initialize everything on mount
  useEffect(() => {
    initializeCamera();
    initializeSpeechRecognition();
    
    return () => {
      // Cleanup
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current);
      }
      if (synthRef.current) {
        synthRef.current.cancel();
      }
    };
  }, [initializeCamera, initializeSpeechRecognition]);

  return (
    <div className="bujji-app">
      <div className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <Eye className="logo-icon" size={32} />
            <h1>Bujji</h1>
            <Badge variant="secondary" className="ai-badge">AI Vision Assistant</Badge>
          </div>
          
          <div className="status-indicators">
            <Badge variant={isScanning ? "default" : "outline"} className="status-badge">
              {isScanning ? "Scanning Active" : "Scanning Paused"}
            </Badge>
            {isAnalyzing && (
              <Badge variant="secondary" className="status-badge">
                <Loader2 className="animate-spin" size={12} />
                Analyzing...
              </Badge>
            )}
          </div>
        </div>
      </div>

      <div className="main-content">
        {/* Video Section */}
        <Card className="video-section">
          <div className="video-container">
            {cameraError ? (
              <div className="camera-error">
                <AlertTriangle size={48} />
                <p>{cameraError}</p>
                <Button onClick={initializeCamera} className="retry-button">
                  <Camera size={16} />
                  Retry Camera
                </Button>
              </div>
            ) : (
              <>
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className="video-feed"
                />
                
                {currentDescription && (
                  <div className="caption-overlay">
                    <div className="caption-content">
                      <Eye size={16} />
                      <span>{currentDescription}</span>
                    </div>
                  </div>
                )}
              </>
            )}
            
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>
        </Card>

        {/* Controls Section */}
        <Card className="controls-section">
          <div className="control-group">
            <h3>Vision Controls</h3>
            <div className="button-group">
              <Button
                onClick={toggleScanning}
                variant={isScanning ? "destructive" : "default"}
                size="lg"
                disabled={!!cameraError}
              >
                <Eye size={20} />
                {isScanning ? "Stop Scanning" : "Start Scanning"}
              </Button>
              
              <Button
                onClick={analyzeScene}
                variant="outline"
                size="lg"
                disabled={isAnalyzing || !!cameraError}
              >
                {isAnalyzing ? (
                  <Loader2 className="animate-spin" size={20} />
                ) : (
                  <Camera size={20} />
                )}
                Analyze Now
              </Button>
            </div>
          </div>

          <div className="control-group">
            <h3>Voice Controls</h3>
            <div className="button-group">
              <Button
                onClick={toggleListening}
                variant={isListening ? "destructive" : "default"}
                size="lg"
                disabled={isProcessingQuestion}
              >
                {isListening ? <MicOff size={20} /> : <Mic size={20} />}
                {isListening ? "Stop Listening" : "Ask Question"}
              </Button>
              
              <Button
                onClick={stopSpeaking}
                variant="outline"
                size="lg"
                disabled={!isSpeaking}
              >
                {isSpeaking ? <VolumeX size={20} /> : <Volume2 size={20} />}
                {isSpeaking ? "Stop Speaking" : "Speak"}
              </Button>
            </div>
          </div>
        </Card>

        {/* Conversation Section */}
        <Card className="conversation-section">
          <h3>
            <MessageSquare size={20} />
            Recent Conversation
          </h3>
          
          <div className="conversation-content">
            {lastQuestion && (
              <div className="conversation-item user-question">
                <Badge variant="outline" className="conversation-label">You asked:</Badge>
                <p>"{lastQuestion}"</p>
              </div>
            )}
            
            {isProcessingQuestion ? (
              <div className="conversation-item processing">
                <Loader2 className="animate-spin" size={16} />
                <span>Thinking about your question...</span>
              </div>
            ) : lastAnswer ? (
              <div className="conversation-item assistant-response">
                <Badge variant="default" className="conversation-label">Bujji:</Badge>
                <p>{lastAnswer}</p>
              </div>
            ) : null}
          </div>
        </Card>
      </div>

      <Toaster />
    </div>
  );
}

export default BujjiVisionAssistant;