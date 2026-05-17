#!/usr/bin/env python3
"""
FastAPI frontend for the Hotel Guest Register OCR Pipeline.
Supports single-image testing and bulk processing.
"""

import os
import json
import base64
import mimetypes
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import requests
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
app = FastAPI(title="OCR Pipeline Frontend")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

API_MODE = os.getenv("API_MODE", "openrouter").strip().lower()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = "gemini-2.5-flash"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Prompt template
EXTRACTION_PROMPT = (
    "You are an expert OCR data extraction assistant. "
    "Analyze this hotel guest register page image and extract every row into a JSON array. "
    "Each object in the array must have these keys exactly: "
    "date, name, age, mobile_no, nationality, permanent_address, from_where_lodger_arrived, "
    "date_and_time_of_arrival, reason_of_visit, confidence. "
    "The confidence value must be one of: low, medium, high. "
    "IMPORTANT: If the mobile_no value contains a comma (',') or appears as a placeholder/ditto mark (meaning the same mobile number as the previous row), DO NOT extract that row. Skip the entire row completely. "
    "Return ONLY the raw JSON array with no markdown formatting, no code fences, and no extra text."
)

COLUMNS = [
    "date",
    "name",
    "age",
    "mobile_no",
    "nationality",
    "permanent_address",
    "from_where_lodger_arrived",
    "date_and_time_of_arrival",
    "reason_of_visit",
    "confidence",
]

# ─── Helper functions ─────────────────────────────────────────────────────────


def encode_image(image_path: Path) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime not in ("image/jpeg", "image/png"):
        img = Image.open(image_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def parse_json_response(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def call_google_api(b64: str, mime: str) -> dict:
    url = f"{GOOGLE_BASE_URL}/models/{GOOGLE_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": EXTRACTION_PROMPT},
                    {"inlineData": {"mimeType": mime, "data": b64}},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
        },
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usageMetadata", {})
    tokens_in = usage.get("promptTokenCount", 0)
    tokens_out = usage.get("candidatesTokenCount", 0)
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("No candidates in Google API response")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    return {"text": text, "tokens_in": tokens_in, "tokens_out": tokens_out}


def call_openrouter_api(b64: str, mime: str) -> dict:
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
        "temperature": 0.1,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No choices in OpenRouter API response")
    text = choices[0].get("message", {}).get("content", "")
    return {"text": text, "tokens_in": tokens_in, "tokens_out": tokens_out}


def process_single_image(image_path: Path) -> dict:
    b64, mime = encode_image(image_path)
    if API_MODE == "google":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set")
        result = call_google_api(b64, mime)
    else:
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set")
        result = call_openrouter_api(b64, mime)

    records = parse_json_response(result["text"])
    if not isinstance(records, list):
        raise ValueError("Parsed JSON is not a list")

    clean_records = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
            
        # Skip row if mobile_no contains a comma or ditto mark
        mobile_val = rec.get("mobile_no", "")
        if mobile_val and isinstance(mobile_val, str):
            v_clean = mobile_val.strip()
            if "," in v_clean or v_clean == "„" or v_clean == '"' or v_clean == "''":
                continue
            
        clean_records.append({col: rec.get(col, "") for col in COLUMNS})

    return {
        "records": clean_records,
        "tokens_in": result["tokens_in"],
        "tokens_out": result["tokens_out"],
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    model = OPENROUTER_MODEL if API_MODE == "openrouter" else GOOGLE_MODEL
    return templates.TemplateResponse("index.html", {"request": request, "api_mode": API_MODE, "model": model})


@app.get("/single", response_class=HTMLResponse)
def single_page(request: Request):
    model = OPENROUTER_MODEL if API_MODE == "openrouter" else GOOGLE_MODEL
    return templates.TemplateResponse("single.html", {"request": request, "api_mode": API_MODE, "model": model})


@app.get("/bulk", response_class=HTMLResponse)
def bulk_page(request: Request):
    model = OPENROUTER_MODEL if API_MODE == "openrouter" else GOOGLE_MODEL
    return templates.TemplateResponse("bulk.html", {"request": request, "api_mode": API_MODE, "model": model})


@app.post("/api/single")
def api_single(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image provided")

    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / image.filename
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        result = process_single_image(temp_path)
        
        # Build Excel
        from openpyxl import Workbook
        from datetime import datetime
        wb = Workbook()
        ws = wb.active
        ws.title = "Extracted"
        headers = ["source_image"] + COLUMNS
        ws.append(headers)
        for rec in result["records"]:
            ws.append([image.filename] + [rec.get(h, "") for h in COLUMNS])
            
        download_dir = Path(tempfile.gettempdir()) / "ocr_downloads"
        download_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_path = download_dir / f"single_extracted_{timestamp}.xlsx"
        wb.save(download_path)
        
        result["download_url"] = f"/download/{download_path.name}"
        
        return {"success": True, **result}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": str(exc)})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/api/bulk")
def api_bulk(images: list[UploadFile] = File(...)):
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")

    temp_dir = Path(tempfile.mkdtemp())
    all_records = []
    total_tokens_in = 0
    total_tokens_out = 0
    errors = []
    processed_count = 0

    try:
        for image in images:
            if not image.filename:
                continue
            temp_path = temp_dir / image.filename
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(image.file, f)
            try:
                result = process_single_image(temp_path)
                for rec in result["records"]:
                    rec["source_image"] = image.filename
                all_records.extend(result["records"])
                total_tokens_in += result["tokens_in"]
                total_tokens_out += result["tokens_out"]
                processed_count += 1
            except Exception as exc:
                errors.append({"file": image.filename, "error": str(exc)})
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        # Build Excel
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Extracted"
        headers = ["source_image"] + COLUMNS
        ws.append(headers)
        for rec in all_records:
            ws.append([rec.get(h, "") for h in headers])

        download_dir = Path(tempfile.gettempdir()) / "ocr_downloads"
        download_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_path = download_dir / f"bulk_extracted_{timestamp}.xlsx"
        wb.save(download_path)

        return {
            "success": True,
            "total_images": len(images),
            "processed": processed_count,
            "errors": errors,
            "total_records": len(all_records),
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "download_url": f"/download/{download_path.name}",
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/download/{filename}")
def download_file(filename: str):
    download_dir = Path(tempfile.gettempdir()) / "ocr_downloads"
    file_path = download_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"Starting OCR Frontend on http://127.0.0.1:8000")
    print(f"API Mode: {API_MODE}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
