# Requirements file
# fastapi
# uvicorn
# openai-whisper
# python-multipart
# pydantic


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
import shutil
from typing import Optional
import logging
from pydantic import BaseModel
import uuid # Import uuid for unique filenames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voice-to-Text API",
    description="Convert audio recordings to text using OpenAI Whisper",
    version="1.0.0"
)

# Response model
class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None

# Suppress the FP16 warning on CPU
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load Whisper model (you can change model size: tiny, base, small, medium, large)
MODEL_SIZE = "base"  # Good balance between speed and accuracy
model = None

@app.on_event("startup")
async def load_model():
    """Load Whisper model on startup"""
    global model
    try:
        logger.info(f"Loading Whisper model: {MODEL_SIZE}")
        # <<< FIX: Explicitly set the device to "cpu" like in your working code
        model = whisper.load_model(MODEL_SIZE, device="cpu")
        logger.info(f"Whisper model loaded successfully on device: {model.device}")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Voice-to-Text API is running", "model": MODEL_SIZE}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None
):
    """
    Transcribe audio file to text

    Args:
        file: Audio file (supported formats: mp3, mp4, wav, m4a, etc.)
        language: Optional language code (e.g., 'en', 'es', 'fr')

    Returns:
        TranscriptionResponse with transcribed text
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Whisper model not loaded")

    # Use a temporary file to store the uploaded audio
    temp_file_path = None
    try:
        # Create a temporary file with the correct extension
        file_extension = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        logger.info(f"Processing file: {file.filename} at {temp_file_path}")

        # Transcribe the audio file
        # No need for fp16=False since we've loaded the model on CPU
        result = model.transcribe(temp_file_path, language=language)

        transcribed_text = result["text"].strip()
        detected_language = result.get("language")

        logger.info(f"Transcription completed. Language: {detected_language}")

        return TranscriptionResponse(
            text=transcribed_text,
            language=detected_language
        )

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Successfully deleted temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {temp_file_path} - {e}")

if __name__ == "__main__":
    import uvicorn
    # To run this, you need uvicorn: pip install "uvicorn[standard]"
    uvicorn.run(app, host="0.0.0.0", port=8000)
