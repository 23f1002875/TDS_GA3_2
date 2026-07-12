import os
import base64
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

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

        # Handle both raw base64 and data URLs
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)

        # Detect actual image format
        img = Image.open(BytesIO(image_bytes))
        mime_type = Image.MIME.get(img.format, "image/png")

        prompt = f"""
You are an expert at reading scanned documents, invoices, receipts,
bar charts, pie charts, tables and forms.

Question:
{req.question}

Instructions:
- Return ONLY the final answer.
- Do NOT explain your reasoning.
- Do NOT include labels or extra text.
- If the answer is a number, return only the number.
- Remove currency symbols.
- Remove commas from numbers.
- Remove units.
- Preserve decimal places if present.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                prompt,
            ],
        )

        answer = response.text.strip()

        return {
            "answer": str(answer)
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to process image."
        )
