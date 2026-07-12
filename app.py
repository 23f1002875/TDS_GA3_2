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


class ImageRequest(BaseModel):
    image_base64: str
    question: str


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/answer-image")
def answer_image(req: ImageRequest):
    try:
        image_data = req.image_base64.strip()

        # Support both raw base64 and data URLs
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png"
                ),
                f"""
Question: {req.question}

Answer ONLY the requested value.

Rules:
- Return only the answer.
- Do not explain.
- If the answer is numeric, return only the number.
- Do not include currency symbols, commas, units, or extra words.
"""
            ]
        )

        answer = response.text.strip()

        return {
            "answer": str(answer)
        }

    except Exception as e:
        return {
            "answer": str(e)
        }
