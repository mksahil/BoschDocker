from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel



app = FastAPI(title="Azure Speech-to-Text API")

AZURE_ENDPOINT = "https://bgsw-openaiservice-voiceseatbooking-p-eus-001.cognitiveservices.azure.com/speechtotext/transcriptions:transcribe"
AZURE_API_KEY = "2u2cSvJIlkFgj6BKsabPTVeIS4zcFlCu49yk2JxzrmUkIDTycp9qJQQJ99BHACYeBjFXJ3w3AAABACOGoOsh"
API_VERSION = "2024-11-15"


class TranscriptionResponse(BaseModel):
    text: str


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...)
):
    """
    Transcribe audio file using Azure Speech-to-Text API
    
    Args:
        audio: Audio file (mp3, wav, etc.)
        locale: Language locale (default: en-IN)
    
    Returns:
        Transcribed text from the audio
    """
    try:
        # Read the uploaded audio file
        audio_content = await file.read()
        
        # Prepare the request
        url = f"{AZURE_ENDPOINT}?api-version={API_VERSION}"
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_API_KEY
        }
        
        # Prepare the definition JSON
        definition = {
            "locales": ["en-IN"]
        }
        
        # Prepare multipart form data
        files = {
            "audio": (file.filename, audio_content, file.content_type),
            "definition": (None, str(definition).replace("'", '"'))
        }
        
        # Make request to Azure API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, files=files)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Azure API error: {response.text}"
                )
            
            # Parse response
            result = response.json()
            
            # Extract only the text from combinedPhrases
            if "combinedPhrases" in result and len(result["combinedPhrases"]) > 0:
                transcribed_text = result["combinedPhrases"][0].get("text", "")
                return JSONResponse(content={"text": transcribed_text})
            else:
                return JSONResponse(content={"text": ""})
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to Azure API timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

