from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import base64

app = FastAPI()

# Load model (lightweight)
tts = pipeline("text-to-speech", model="parler-tts/parler_tts_mini")

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "TTS Server Running 🚀"}

@app.post("/tts")
def generate_tts(input: TextInput):
    result = tts(input.text)

    audio_bytes = result["audio"]

    # Convert to base64 (important for API)
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "audio": audio_base64
    }
