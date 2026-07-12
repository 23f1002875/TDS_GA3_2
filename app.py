import os
import base64

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


class ImageQuestion(BaseModel):
    image_base64: str
    question: str


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/answer-image")
async def answer_image(data: ImageQuestion):

    image_bytes = base64.b64decode(data.image_base64)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            ),
            data.question,
        ],
    )

    return {
        "answer": response.text.strip()
    }
