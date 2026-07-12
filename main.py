import json, re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import config

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)
HEAD = {"Authorization": f"Bearer {config.AIPIPE_TOKEN}", "Content-Type": "application/json"}

async def chat(messages, model=None, max_tokens=1200):
    body = {"model": model or config.TEXT_MODEL, "messages": messages,
            "temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}}
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(f"{config.AIPIPE_BASE}/chat/completions", headers=HEAD, json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

def parse_json(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?|\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        return json.loads(m.group(0)) if m else {}

def normalize_answer(ans):
    s = str(ans).strip()
    if not s:
        return s
    cleaned = re.sub(r"[,\s]", "", s)
    cleaned = re.sub(r"[₹$€£%]", "", cleaned)
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if m and re.fullmatch(r"[^\dA-Za-z]*-?\d[\d,.\s₹$€£%]*", s.strip()):
        num = m.group(0)
        if "." in num:
            num = num.rstrip("0").rstrip(".")
        return num
    return s

@app.get("/")
async def root():
    return {"ok": True, "email": config.EMAIL}

@app.post("/answer-image")
async def answer_image(request: Request):
    body = await request.json()
    img_b64 = body.get("image_base64", "")
    question = body.get("question", "")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text":
                "You read charts, receipts, tables, invoices and pie charts EXACTLY.\n"
                "1. Transcribe every relevant label/number carefully.\n"
                "2. If arithmetic is needed, compute it step by step and double-check.\n"
                "3. Final 'answer': if numeric, output ONLY the bare number, no symbols/units. "
                "If text, output exactly as written in the image.\n"
                "Return JSON: {\"work\": \"...\", \"answer\": \"...\"}.\n"
                f"Question: {question}"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
        ],
    }]
    try:
        out = parse_json(await chat(messages, model=config.VISION_MODEL))
        ans = normalize_answer(out.get("answer", ""))
    except Exception:
        ans = ""
    return {"answer": str(ans)}
