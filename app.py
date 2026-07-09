import base64
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
def answer_image(req: ImageRequest):
    image_bytes = base64.b64decode(req.image_base64)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            req.question,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            ),
        ],
    )

    return {
        "answer": response.text.strip()
    }
