#!/usr/bin/env python3
"""
Hotel Guest Register OCR Extraction Pipeline
Extracts structured guest data from handwritten hotel register page images using Vision LLMs.
Supports Google AI Studio and OpenRouter APIs.
"""

import os
import sys
import json
import base64
import logging
import mimetypes
from pathlib import Path
from datetime import datetime

import requests
from PIL import Image
from openpyxl import Workbook
from dotenv import load_dotenv

# ─── Hardcoded pricing (per 1M tokens, in USD) ──────────────────────────────
# Adjust these constants at the top of the file as needed.
PRICE_INPUT = {
    "google/gemini-2.5-flash": 0.15,
    "google/gemini-3-flash-preview": 0.10,
}
PRICE_OUTPUT = {
    "google/gemini-2.5-flash": 0.60,
    "google/gemini-3-flash-preview": 0.40,
}

# ─── Configuration ────────────────────────────────────────────────────────────
load_dotenv()

API_MODE = os.getenv("API_MODE", "google").strip().lower()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = "gemini-2.5-flash"
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SAMPLE_IMAGES_DIR = Path("sample_images")
OUTPUT_EXCEL = Path("output_extracted.xlsx")
FLAGGED_EXCEL = Path("flagged_review.xlsx")
ERROR_LOG = Path("errors.log")
LIMIT_IMAGES = None  # Set to an integer to limit, or None to process all images

# Prompt template
EXTRACTION_PROMPT = (
    "You are an expert OCR data extraction assistant. "
    "Analyze this hotel guest register page image and extract every row into a JSON array. "
    "Each object in the array must have these keys exactly: "
    "date, name, age, mobile_no, nationality, permanent_address, from_where_lodger_arrived, "
    "date_and_time_of_arrival, reason_of_visit, confidence. "
    "The confidence value must be one of: low, medium, high. "
    "Return ONLY the raw JSON array with no markdown formatting, no code fences, and no extra text."
)

# Logging
logging.basicConfig(
    filename=ERROR_LOG,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)

# ─── Helper functions ─────────────────────────────────────────────────────────


def encode_image(image_path: Path) -> tuple[str, str]:
    """Return (base64_encoded_string, mime_type) for the image."""
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime not in ("image/jpeg", "image/png"):
        # Try to open with Pillow and re-encode as JPEG for compatibility
        img = Image.open(image_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def parse_json_response(text: str) -> list[dict]:
    """Clean and parse JSON from model response text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def parse_date(date_str: str) -> datetime | None:
    """Try to parse a date string into a datetime object."""
    if not date_str or not isinstance(date_str, str):
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def call_google_api(image_path: Path, b64: str, mime: str) -> dict:
    """Call Google Generative Language API."""
    url = f"{GOOGLE_BASE_URL}/models/{GOOGLE_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": EXTRACTION_PROMPT},
                    {
                        "inlineData": {
                            "mimeType": mime,
                            "data": b64,
                        }
                    },
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


def call_openrouter_api(image_path: Path, b64: str, mime: str) -> dict:
    """Call OpenRouter chat completions API."""
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
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
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


def process_images() -> None:
    if not SAMPLE_IMAGES_DIR.exists():
        print(f"Error: Image directory not found: {SAMPLE_IMAGES_DIR}")
        sys.exit(1)

    image_files = sorted(
        [
            p
            for p in SAMPLE_IMAGES_DIR.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
    )
    if not image_files:
        print(f"No JPEG/PNG images found in {SAMPLE_IMAGES_DIR}")
        sys.exit(1)

    if LIMIT_IMAGES:
        image_files = image_files[:LIMIT_IMAGES]

    all_records: list[dict] = []
    flagged_records: list[dict] = []
    total_tokens_in = 0
    total_tokens_out = 0
    total_images = len(image_files)
    processed_images = 0
    error_count = 0

    wb_out = Workbook()
    ws_out = wb_out.active
    ws_out.title = "Extracted"
    headers = [
        "source_image",
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
    ws_out.append(headers)

    wb_flag = Workbook()
    ws_flag = wb_flag.active
    ws_flag.title = "Flagged"
    flag_headers = headers + ["flag_reason"]
    ws_flag.append(flag_headers)

    print(f"Processing {total_images} image(s) using API_MODE={API_MODE}...")

    for img_path in image_files:
        print(f"  -> {img_path.name}")
        try:
            b64, mime = encode_image(img_path)
            if API_MODE == "google":
                if not GOOGLE_API_KEY:
                    raise ValueError("GOOGLE_API_KEY is not set")
                result = call_google_api(img_path, b64, mime)
            else:
                if not OPENROUTER_API_KEY:
                    raise ValueError("OPENROUTER_API_KEY is not set")
                result = call_openrouter_api(img_path, b64, mime)

            text = result["text"]
            total_tokens_in += result["tokens_in"]
            total_tokens_out += result["tokens_out"]

            records = parse_json_response(text)
            if not isinstance(records, list):
                raise ValueError("Parsed JSON is not a list")

            for rec in records:
                if not isinstance(rec, dict):
                    continue
                row = {
                    "source_image": img_path.name,
                    "date": rec.get("date", ""),
                    "name": rec.get("name", ""),
                    "age": rec.get("age", ""),
                    "mobile_no": rec.get("mobile_no", ""),
                    "nationality": rec.get("nationality", ""),
                    "permanent_address": rec.get("permanent_address", ""),
                    "from_where_lodger_arrived": rec.get("from_where_lodger_arrived", ""),
                    "date_and_time_of_arrival": rec.get("date_and_time_of_arrival", ""),
                    "reason_of_visit": rec.get("reason_of_visit", ""),
                    "confidence": rec.get("confidence", ""),
                }
                all_records.append(row)

                reasons = []
                if not str(row["name"]).strip():
                    reasons.append("empty_name")
                if str(row.get("confidence", "")).strip().lower() == "low":
                    reasons.append("low_confidence")

                if reasons:
                    row_copy = dict(row)
                    row_copy["flag_reason"] = ", ".join(reasons)
                    flagged_records.append(row_copy)

            processed_images += 1

        except Exception as exc:
            error_count += 1
            msg = f"{img_path.name}: {exc}"
            logging.error(msg)
            print(f"     ERROR: {exc}")
            continue

    for rec in all_records:
        ws_out.append([rec[h] for h in headers])
    wb_out.save(OUTPUT_EXCEL)

    for rec in flagged_records:
        ws_flag.append([rec.get(h, "") for h in flag_headers])
    wb_flag.save(FLAGGED_EXCEL)

    total_records = len(all_records)
    flagged_count = len(flagged_records)
    pct_flagged = (flagged_count / total_records * 100) if total_records else 0.0

    model_key = GOOGLE_MODEL if API_MODE == "google" else OPENROUTER_MODEL
    input_rate = PRICE_INPUT.get(model_key, 0.0)
    output_rate = PRICE_OUTPUT.get(model_key, 0.0)
    est_cost = (total_tokens_in / 1_000_000 * input_rate) + (
        total_tokens_out / 1_000_000 * output_rate
    )

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"{'Images processed':<22} : {processed_images}")
    print(f"{'Images with errors':<22} : {error_count}")
    print(f"{'Total records':<22} : {total_records}")
    print(f"{'Flagged records':<22} : {flagged_count}")
    print(f"{'% Flagged':<22} : {pct_flagged:.2f}%")
    print(f"{'Input tokens':<22} : {total_tokens_in}")
    print(f"{'Output tokens':<22} : {total_tokens_out}")
    print(f"{'Estimated cost (USD)':<22} : ${est_cost:.6f}")
    print(f"{'Output file':<22} : {OUTPUT_EXCEL}")
    print(f"{'Flagged file':<22} : {FLAGGED_EXCEL}")
    if error_count:
        print(f"{'Error log':<22} : {ERROR_LOG}")
    print("=" * 60)


if __name__ == "__main__":
    process_images()
