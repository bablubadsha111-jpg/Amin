from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import base64

app = FastAPI()

tts = None  # initially empty

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Server Running ✅"}

@app.post("/tts")
def generate_tts(input: TextInput):
    global tts
    
    if tts is None:
        tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

    result = tts(input.text)
    audio_bytes = result["audio"]

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {"audio": audio_base64}
