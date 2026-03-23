from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import base64
import traceback

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ CORS enable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts = None

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Server Running ✅"}

@app.post("/tts")
def generate_tts(input: TextInput):
    global tts

    try:
        print("📥 Text received:", input.text)

        # ✅ load model only once
        if tts is None:
            print("⏳ Loading model...")
            tts = pipeline(
                "text-to-speech",
                model="facebook/mms-tts-eng"
            )
            print("✅ Model loaded")

        # ✅ generate audio
        result = tts(input.text)
        audio_bytes = result["audio"]

        # ✅ convert to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        print("✅ Audio generated")

        return {
            "audio": audio_base64
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        print(traceback.format_exc())

        return {
            "error": str(e),
            "message": "TTS generation failed"
        }
