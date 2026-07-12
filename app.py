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
    try:

        image = data.image_base64.strip()

        mime = "image/png"

        if image.startswith("data:"):
            header, image = image.split(",", 1)

            if "jpeg" in header or "jpg" in header:
                mime = "image/jpeg"
            elif "webp" in header:
                mime = "image/webp"
            elif "png" in header:
                mime = "image/png"

        image_bytes = base64.b64decode(image)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction="""
You are a precise document-reading AI.

Read charts, invoices, receipts, tables and scanned documents.

Rules:
- Answer ONLY the user's question.
- Never explain.
- Never add extra words.
- If the answer is numeric, output only the number.
- Remove currency symbols.
- Remove commas from numbers.
- Do not include units unless explicitly asked.
- Preserve names exactly.
- If multiple values exist, choose the correct one based on the question.
"""
            ),
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime,
                ),
                data.question,
            ],
        )

        answer = (response.text or "").strip()

        answer = answer.replace("$", "")
        answer = answer.replace("₹", "")
        answer = answer.replace(",", "")
        answer = answer.replace('"', "")

        return {
            "answer": answer
        }

    except Exception as e:
        print(e)
        return {
            "answer": ""
        }
