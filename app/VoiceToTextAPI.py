from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import shutil
from typing import Optional
import logging
from pydantic import BaseModel
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voice-to-Text API with Azure OpenAI Whisper",
    description="Convert audio recordings to text using the Azure OpenAI Whisper model",
    version="1.0.0"
)

# Response model
class TranscriptionResponse(BaseModel):
    text: str

# Azure OpenAI Configuration
# It's highly recommended to use environment variables for these
AZURE_OPENAI_API_KEY ="2ZZSQcyYdJ8BYy8V9DvJdN6b301z9v3SCY5lOKd0v0nl7XST8264JQQJ99BCACfhMk5XJ3w3AAAAACOG6GMb"
AZURE_OPENAI_ENDPOINT = "https://mkath-m8h24zm2-swedencentral.cognitiveservices.azure.com/openai/deployments/whisper/audio/translations?api-version=2024-06-01"
WHISPER_DEPLOYMENT_NAME = "whisper"

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, WHISPER_DEPLOYMENT_NAME]):
    raise RuntimeError("AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and WHISPER_DEPLOYMENT_NAME environment variables must be set.")

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-09-01-preview",  # Use an appropriate API version
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Voice-to-Text API with Azure OpenAI Whisper is running"}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...)
):
    """
    Transcribe audio file to text using Azure OpenAI Whisper, optimized for Indian English.

    Args:
        file: Audio file (supported formats: mp3, mp4, wav, m4a, etc.)

    Returns:
        TranscriptionResponse with transcribed text
    """
    temp_file_path = None
    try:
        file_extension = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        logger.info(f"Processing file: {file.filename} at {temp_file_path}")

        with open(temp_file_path, "rb") as audio_file:
            
            # This prompt is key to getting the desired behavior.
            # It tells the model to expect Indian English and how to handle proper nouns.
            transcription_prompt = (
                "The following audio is in Indian English. It may contain proper nouns, "
                "place names, or other local terms. Transcribe these names phonetically "
                "using English letters. Do not translate them or convert them to other English words."
            )
            
            result = client.audio.transcriptions.create(
                model=WHISPER_DEPLOYMENT_NAME,
                file=audio_file,
                # 1. Add the detailed prompt to guide the model
                prompt=transcription_prompt,
                # 2. Set temperature to a low value for more predictable, phonetic results
                temperature=0.1 
            )

        transcribed_text = result.text.strip()
        logger.info("Transcription successful.")

        return TranscriptionResponse(text=transcribed_text)

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
