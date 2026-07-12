import json
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import config

app = FastAPI()

# CORS wide open — grader calls from a Cloudflare Worker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

HEAD = {
    "Authorization": f"Bearer {config.AIPIPE_TOKEN}",
    "Content-Type": "application/json",
}


async def chat(messages, model=None, max_tokens=1200, retries=3):
    """Call AIPipe's OpenAI-compatible chat endpoint with basic retry on transient errors."""
    body = {
        "model": model or config.TEXT_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    last_err = None
    async with httpx.AsyncClient(timeout=90) as c:
        for attempt in range(retries):
            try:
                r = await c.post(f"{config.AIPIPE_BASE}/chat/completions", headers=HEAD, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {r.text[:200]}"
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = str(e)
    raise RuntimeError(f"chat failed after {retries} attempts: {last_err}")


def parse_json(s):
    """Strip markdown fences if present and parse JSON, falling back to regex extraction."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


def normalize_numeric(ans):
    """
    If the answer contains a number, extract and return the cleaned bare number
    as a string. Otherwise return None (caller should treat it as text).
    """
    s = str(ans).strip()
    if not s:
        return None
    cleaned = re.sub(r"[,\s]", "", s)
    cleaned = re.sub(r"[₹$€£%]", "", cleaned)
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    # Everything left after removing the numbers — if there's real alphabetic
    # text remaining (more than a currency/unit word), treat this as NOT numeric.
    remainder = re.sub(r"-?\d+(?:\.\d+)?", "", cleaned)
    if re.search(r"[A-Za-z]{2,}", remainder):
        return None
    # Prefer the last number found (handles "Total: 4089.35" style leftovers)
    num = matches[-1]
    if "." in num:
        num = num.rstrip("0").rstrip(".")
        if num == "" or num == "-":
            num = "0"
    return num


def clean_text_answer(s):
    """For non-numeric answers (e.g. pie chart category names), strip trailing
    percentage/parenthetical annotations the model sometimes adds."""
    s = str(s).strip()
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()
    s = re.sub(r"\s*[-–:]\s*\d+(\.\d+)?%?\s*$", "", s).strip()
    return s


last_debug_info = {}


@app.get("/")
async def root():
    return {"ok": True, "email": config.EMAIL}


@app.get("/debug")
async def get_debug():
    """After a failed grader Check, open https://<your-url>.onrender.com/debug
    in a browser to see exactly what the model was asked, what it saw, and what
    it returned for the LAST call. This tells you if it's a reading error, a
    math error, or a formatting error."""
    return last_debug_info


@app.post("/answer-image")
async def answer_image(request: Request):
    global last_debug_info
    body = await request.json()
    img_b64 = body.get("image_base64", "")
    question = body.get("question", "")

    # Guard against an already-prefixed data URL being sent
    if img_b64.startswith("data:image"):
        img_b64 = img_b64.split(",", 1)[-1]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You read charts, receipts, tables, invoices and pie charts EXACTLY.\n"
                        "1. TRANSCRIBE every relevant label and number you see, one by one "
                        "(e.g. each bar's value, each line item, each pie slice %). Read "
                        "digits carefully; do not round or estimate. Put every raw number "
                        "you transcribe into a JSON array called 'values'.\n"
                        "2. Answer the question directly in an 'answer' field.\n"
                        "3. If the final answer is NUMERIC, 'answer' should be ONLY the "
                        "bare number — no currency symbol, no thousands separators, no "
                        "units, no words. Keep decimals exactly as shown in the image.\n"
                        "4. If the final answer is TEXT (e.g. a category name), output it "
                        "EXACTLY as written in the image — no extra words, no percentages.\n"
                        "Return ONLY JSON: {\"values\": [num1, num2, ...], \"work\": \"...\", "
                        "\"answer\": \"...\"}.\n"
                        f"Question: {question}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"},
                },
            ],
        }
    ]

    last_debug_info = {
        "question": question,
        "img_b64_len": len(img_b64),
        "img_b64_prefix": img_b64[:30],
    }

    try:
        raw = await chat(messages, model=config.VISION_MODEL, max_tokens=1200)
        last_debug_info["raw_model_output"] = raw
        out = parse_json(raw)
        last_debug_info["parsed_json"] = out

        values = out.get("values", [])
        raw_answer = out.get("answer", "")
        q_lower = question.lower()

        ans = None

        # For sum/total-style questions, compute the arithmetic ourselves
        # instead of trusting the model's mental math.
        if any(w in q_lower for w in ["total", "sum", "combined", "altogether", "overall"]) and values:
            try:
                nums = [float(str(v).replace(",", "")) for v in values]
                total = sum(nums)
                total = int(total) if float(total).is_integer() else round(total, 2)
                ans = str(total)
                last_debug_info["computed_from_values"] = True
            except Exception as e:
                last_debug_info["sum_error"] = str(e)
                ans = None

        if ans is None:
            numeric = normalize_numeric(raw_answer)
            ans = numeric if numeric is not None else clean_text_answer(raw_answer)

        last_debug_info["final_answer"] = ans

    except Exception as e:
        last_debug_info["exception"] = str(e)
        ans = ""

    return {"answer": str(ans)}
