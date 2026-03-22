from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import base64

# ✅ CORS import
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ CORS enable (VERY IMPORTANT for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sab allow (testing ke liye)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts = None  # model initially empty

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Server Running ✅"}

@app.post("/tts")
def generate_tts(input: TextInput):
    global tts

    try:
        # ✅ model load only once
        if tts is None:
            tts = pipeline(
                "text-to-speech",
                model="facebook/mms-tts-eng"  # lightweight model
            )

        # ✅ generate audio
        result = tts(input.text)
        audio_bytes = result["audio"]

        # ✅ convert to base64 string
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {"audio": audio_base64}

    except Exception as e:
        # ✅ error handle (important for debugging)
        return {
            "error": str(e),
            "message": "TTS generation failed"
        }
