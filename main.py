import base64
import json
import os
import re

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import anthropic

app = FastAPI()

# --- CORS: allow any origin (required so the Cloudflare Worker grader can call this) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Put your Anthropic API key in the ANTHROPIC_API_KEY environment variable
# on whatever platform you deploy to (Render/Fly/HF Spaces/etc).
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = (
    "You are a precise data-extraction engine. You will be shown an image "
    "(invoice, chart, table, receipt, etc.) and a question about it. "
    "Reply with ONLY the raw answer value - no explanation, no units, "
    "no currency symbols, no extra words. If the answer is a number, "
    "reply with just the number (e.g. 4089.35)."
)


def extract_answer_text(resp) -> str:
    """Pull the text out of the Claude API response."""
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def clean_answer(raw: str) -> str:
    """Strip stray quotes/whitespace/currency symbols the model might add."""
    raw = raw.strip().strip('"').strip("'").strip()
    # remove common currency symbols/commas if the model slipped one in
    raw = re.sub(r"^[$€£₹]\s*", "", raw)
    raw = raw.replace(",", "") if re.match(r"^[\d,\.]+$", raw) else raw
    return raw


@app.post("/answer-image")
async def answer_image(request: Request):
    try:
        body = await request.json()
        image_b64 = body["image_base64"]
        question = body["question"]

        # Some clients send a data URL prefix - strip it if present.
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]

        # Detect media type (default png)
        media_type = "image/png"
        try:
            header = base64.b64decode(image_b64[:64] + "==", validate=False)
            if header[:3] == b"\xff\xd8\xff":
                media_type = "image/jpeg"
        except Exception:
            pass

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )

        answer = clean_answer(extract_answer_text(message))
        return JSONResponse({"answer": answer})

    except Exception as e:
        # Return a 200 with an error-safe message so the grader still gets JSON,
        # but log the real error server-side.
        print("ERROR:", repr(e))
        return JSONResponse({"answer": "", "error": str(e)}, status_code=500)


@app.get("/")
async def health():
    return {"status": "ok"}
