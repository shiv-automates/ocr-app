#!/usr/bin/env python3
"""
FastAPI frontend for the Hotel Guest Register OCR Pipeline.
Supports single-image testing and bulk processing.
NOW WITH: Automatic Odoo CRM Lead sync.
"""

import os
import json
import base64
import mimetypes
import tempfile
import shutil
import xmlrpc.client
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
app = FastAPI(title="OCR Pipeline Frontend with Odoo CRM Sync")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

API_MODE = os.getenv("API_MODE", "openrouter").strip().lower()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = "gemini-2.5-flash"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ─── Odoo Configuration ───────────────────────────────────────────────────────
ODOO_URL = os.getenv("ODOO_URL", "").rstrip("/")
ODOO_DB = os.getenv("ODOO_DB", "")
ODOO_USERNAME = os.getenv("ODOO_USERNAME", "")
ODOO_API_KEY = os.getenv("ODOO_API_KEY", "")
ODOO_SYNC_ENABLED = os.getenv("ODOO_SYNC_ENABLED", "true").strip().lower() == "true"

ODOO_SOURCE_NAME = "Hotel Register (OCR)"

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


# ─── Odoo Integration ─────────────────────────────────────────────────────────


def odoo_authenticate() -> tuple[int, object]:
    """Authenticate with Odoo and return (uid, models proxy)."""
    if not all([ODOO_URL, ODOO_DB, ODOO_USERNAME, ODOO_API_KEY]):
        raise ValueError("Odoo credentials not fully configured")
    
    common = xmlrpc.client.ServerProxy(f"{ODOO_URL}/xmlrpc/2/common")
    uid = common.authenticate(ODOO_DB, ODOO_USERNAME, ODOO_API_KEY, {})
    if not uid:
        raise ValueError("Odoo authentication failed")
    
    models = xmlrpc.client.ServerProxy(f"{ODOO_URL}/xmlrpc/2/object")
    return uid, models


NATIONALITY_MAP = {
    "ind": "India", "in": "India", "indian": "India", "india": "India",
    "aus": "Australia", "au": "Australia", "australia": "Australia", "australian": "Australia",
    "usa": "United States", "us": "United States", "american": "United States", "united states": "United States",
    "uk": "United Kingdom", "gb": "United Kingdom", "british": "United Kingdom", "britain": "United Kingdom",
    "ger": "Germany", "de": "Germany", "german": "Germany", "germany": "Germany",
    "bel": "Belgium", "be": "Belgium", "belgian": "Belgium", "belgium": "Belgium",
    "fra": "France", "fr": "France", "french": "France", "france": "France",
    "lat": "Latvia", "lv": "Latvia", "latvian": "Latvia", "latvia": "Latvia",
    "nep": "Nepal", "np": "Nepal", "nepali": "Nepal", "nepal": "Nepal",
    "lka": "Sri Lanka", "sl": "Sri Lanka", "srilankan": "Sri Lanka", "sri lanka": "Sri Lanka",
    "pak": "Pakistan", "pk": "Pakistan", "pakistani": "Pakistan", "pakistan": "Pakistan",
    "bgd": "Bangladesh", "bd": "Bangladesh", "bangladeshi": "Bangladesh", "bangladesh": "Bangladesh",
    "chn": "China", "cn": "China", "chinese": "China", "china": "China",
    "jpn": "Japan", "jp": "Japan", "japanese": "Japan", "japan": "Japan",
    "tha": "Thailand", "th": "Thailand", "thai": "Thailand", "thailand": "Thailand",
    "ita": "Italy", "it": "Italy", "italian": "Italy", "italy": "Italy",
    "esp": "Spain", "es": "Spain", "spanish": "Spain", "spain": "Spain",
    "nld": "Netherlands", "nl": "Netherlands", "dutch": "Netherlands", "netherlands": "Netherlands",
    "rus": "Russia", "ru": "Russia", "russian": "Russia", "russia": "Russia",
    "can": "Canada", "ca": "Canada", "canadian": "Canada",
    "nz": "New Zealand", "nzl": "New Zealand", "new zealand": "New Zealand",
}

_country_cache: dict = {}


def resolve_country_id(models: object, uid: int, nationality: str) -> int | None:
    """Look up Odoo country ID from a nationality string (code, name, or adjective)."""
    if not nationality or not nationality.strip():
        return None
    nat = nationality.strip().lower()

    mapped = NATIONALITY_MAP.get(nat, nat)
    mapped_lower = mapped.lower()

    if mapped_lower in _country_cache:
        return _country_cache[mapped_lower]

    for search_val in [mapped, nationality.strip()]:
        ids = models.execute_kw(
            ODOO_DB, uid, ODOO_API_KEY,
            "res.country", "search",
            [[("name", "ilike", search_val)]],
            {"limit": 1}
        )
        if ids:
            _country_cache[mapped_lower] = ids[0]
            return ids[0]

    codes = [nat.upper()]
    if len(nat) == 3:
        codes.append(nat[:2].upper())
    for code in codes:
        ids = models.execute_kw(
            ODOO_DB, uid, ODOO_API_KEY,
            "res.country", "search",
            [[("code", "in", [code])]],
            {"limit": 1}
        )
        if ids:
            _country_cache[mapped_lower] = ids[0]
            return ids[0]

    return None


def get_or_create_odoo_source(models: object, uid: int) -> int:
    """Get or create the UTM source 'Hotel Register (OCR)' in Odoo."""
    domain = [[("name", "=", ODOO_SOURCE_NAME)]]
    source_ids = models.execute_kw(
        ODOO_DB, uid, ODOO_API_KEY,
        "utm.source", "search", domain
    )
    if source_ids:
        return source_ids[0]
    
    # Create new source
    new_id = models.execute_kw(
        ODOO_DB, uid, ODOO_API_KEY,
        "utm.source", "create", [{"name": ODOO_SOURCE_NAME}]
    )
    return new_id


def build_lead_description(rec: dict) -> str:
    """Build a rich description from the OCR record."""
    lines = ["Guest Register Entry"]
    if rec.get("age"):
        lines.append(f"Age: {rec['age']}")
    if rec.get("nationality"):
        lines.append(f"Nationality: {rec['nationality']}")
    if rec.get("permanent_address"):
        lines.append(f"Permanent Address: {rec['permanent_address']}")
    if rec.get("from_where_lodger_arrived"):
        lines.append(f"Arrived From: {rec['from_where_lodger_arrived']}")
    if rec.get("date_and_time_of_arrival"):
        lines.append(f"Arrival: {rec['date_and_time_of_arrival']}")
    if rec.get("reason_of_visit"):
        lines.append(f"Reason: {rec['reason_of_visit']}")
    if rec.get("date"):
        lines.append(f"Register Date: {rec['date']}")
    lines.append(f"Confidence: {rec.get('confidence', 'unknown')}")
    return "\n".join(lines)


def sync_records_to_odoo(records: list[dict]) -> dict:
    """Push records to Odoo CRM. Returns sync summary."""
    summary = {
        "synced_count": 0,
        "skipped_low_confidence": 0,
        "duplicates_skipped": 0,
        "errors": [],
    }

    if not ODOO_SYNC_ENABLED:
        summary["errors"].append("Odoo sync is disabled via ODOO_SYNC_ENABLED")
        return summary

    try:
        uid, models = odoo_authenticate()
        source_id = get_or_create_odoo_source(models, uid)
    except Exception as exc:
        summary["errors"].append(f"Odoo connection failed: {exc}")
        return summary

    for rec in records:
        try:
            # Skip low confidence
            confidence = str(rec.get("confidence", "")).strip().lower()
            if confidence == "low":
                summary["skipped_low_confidence"] += 1
                continue

            name = str(rec.get("name", "")).strip()
            phone = str(rec.get("mobile_no", "")).strip()

            if not name or not phone:
                summary["skipped_low_confidence"] += 1
                continue

            # Deduplication: check if lead with same name + phone exists
            domain = [[
                ("name", "=", name),
                ("phone", "=", phone),
            ]]
            existing = models.execute_kw(
                ODOO_DB, uid, ODOO_API_KEY,
                "crm.lead", "search", domain
            )
            if existing:
                summary["duplicates_skipped"] += 1
                continue

            # Create lead
            lead_vals = {
                "name": name,
                "phone": phone,
                "type": "lead",
                "source_id": source_id,
                "description": build_lead_description(rec),
            }

            # Map address to street if present
            address = str(rec.get("permanent_address", "")).strip()
            if address:
                lead_vals["street"] = address

            # Map nationality to country
            nationality = str(rec.get("nationality", "")).strip()
            if nationality:
                country_id = resolve_country_id(models, uid, nationality)
                if country_id:
                    lead_vals["country_id"] = country_id

            # Map arrival city/place
            arrived_from = str(rec.get("from_where_lodger_arrived", "")).strip()
            if arrived_from:
                lead_vals["city"] = arrived_from

            models.execute_kw(
                ODOO_DB, uid, ODOO_API_KEY,
                "crm.lead", "create", [lead_vals]
            )
            summary["synced_count"] += 1

        except Exception as exc:
            summary["errors"].append(f"Row sync failed for '{rec.get('name', 'unknown')}': {exc}")

    return summary


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
        
        # ─── Odoo Sync ────────────────────────────────────────────────────────
        odoo_summary = sync_records_to_odoo(result["records"])
        result["odoo_sync"] = odoo_summary
        
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

        # ─── Odoo Sync ────────────────────────────────────────────────────────
        odoo_summary = sync_records_to_odoo(all_records)

        return {
            "success": True,
            "total_images": len(images),
            "processed": processed_count,
            "errors": errors,
            "total_records": len(all_records),
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "download_url": f"/download/{download_path.name}",
            "odoo_sync": odoo_summary,
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
    print(f"Odoo Sync Enabled: {ODOO_SYNC_ENABLED}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
