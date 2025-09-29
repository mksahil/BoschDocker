from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
import shutil
from typing import Optional, TypeVar, Generic
import logging
from pydantic import BaseModel, Field
import uuid # Included from your original code

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voice-to-Text API",
    description="Convert audio recordings to text using OpenAI Whisper",
    version="1.0.0"
)

# --- New Pydantic Models for the desired response structure ---

# 1. Define the model for the actual data payload inside "ResponseData"
class TranscriptionData(BaseModel):
    text: str
    language: Optional[str] = None

# 2. Define a generic wrapper model for all API responses
T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    IsSuccess: bool = Field(True, description="Indicates if the request was successful.")
    ErrorMessage: str = Field("", description="Provides an error message if IsSuccess is false.")
    Message: str = Field("Success", description="A general status message.")
    ResponseData: Optional[T] = Field(None, description="The main data payload of the response.")


# --- End of new models ---


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
        model = whisper.load_model(MODEL_SIZE, device="cpu")
        logger.info(f"Whisper model loaded successfully on device: {model.device}")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        # Note: An error here will prevent the app from starting up.
        # This is generally the desired behavior.
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Voice-to-Text API is running", "model": MODEL_SIZE}

@app.post("/transcribe", response_model=ApiResponse[TranscriptionData])
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
        ApiResponse with the transcription result in the ResponseData field.
    """
    if model is None:
        # Instead of raising HTTPException, return a structured JSON response for errors
        error_response = ApiResponse(
            IsSuccess=False,
            ErrorMessage="Whisper model is not available or failed to load.",
            Message="Internal Server Error",
            ResponseData=None
        ).dict()
        return JSONResponse(status_code=500, content=error_response)

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
        result = model.transcribe(temp_file_path, language=language)

        transcribed_text = result["text"].strip()
        detected_language = result.get("language")

        logger.info(f"Transcription completed. Language: {detected_language}")

        # 1. Create the inner data payload
        transcription_data = TranscriptionData(
            text=transcribed_text,
            language=detected_language
        )
        
        # 2. Wrap the data payload in the main ApiResponse model for a successful response
        return ApiResponse[TranscriptionData](ResponseData=transcription_data)

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        # Return the custom error response format
        error_response = ApiResponse(
            IsSuccess=False,
            ErrorMessage=f"Transcription failed: {str(e)}",
            Message="Transcription Error",
            ResponseData=None
        ).dict()
        return JSONResponse(status_code=500, content=error_response)

    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Successfully deleted temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {temp_file_path} - {e}")
