from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import os
import json
import re
from dotenv import load_dotenv
from anthropic import Anthropic

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()

API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not API_KEY:
    raise ValueError("Missing ANTHROPIC_API_KEY in .env file")

client = Anthropic(api_key=API_KEY)

# -----------------------------
# FASTAPI SETUP
# -----------------------------
app = FastAPI(title="FinMap AI + Claude Schema Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# KNOWN SCHEMA ONLY
# -----------------------------
KNOWN_SCHEMA = {
    "Revenue": "revenue",
    "OperatingCost": "operating_expense",
    "DSCR": "dscr"
}

# -----------------------------
# MAP KNOWN COLUMNS
# -----------------------------
def map_known(columns):
    mapped = {}
    unmapped = []

    for col in columns:
        if col in KNOWN_SCHEMA:
            mapped[col] = {
                "mapping": KNOWN_SCHEMA[col],
                "confidence": 1.0,
                "source": "deterministic"
            }
        else:
            unmapped.append(col)

    return mapped, unmapped

# -----------------------------
# JSON EXTRACTION (FIX)
# -----------------------------
def extract_json(text):
    """
    Extract JSON from Claude response.
    Handles markdown-wrapped JSON and messy outputs.
    """
    # Case 1: ```json ... ```
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # Case 2: any {...}
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)

    return text


# -----------------------------
# CLAUDE ANALYSIS
# -----------------------------
def analyze_with_claude(columns):
    if not columns:
        return {}

    # Cleaner formatting for LLM
    columns_str = "\n".join(f"- {col}" for col in columns)

    prompt = f"""
You are analyzing UNKNOWN column names from a financial dataset.

Rules:
- Do NOT assume they map to known schema
- Do NOT force equivalence (e.g., revenue != income unless obvious)
- Provide cautious interpretations
- If unsure, say so

For each column return:
- possible_meaning
- confidence (0 to 1)
- reasoning

Columns:
{columns_str}

Return ONLY raw JSON.
Do NOT wrap in markdown.
Do NOT include backticks.

Format:
{{
  "ColumnName": {{
    "possible_meaning": "...",
    "confidence": 0.0,
    "reasoning": "..."
  }}
}}
"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = response.content[0].text

    try:
        cleaned = extract_json(raw_text)
        return json.loads(cleaned)
    except Exception as e:
        return {
            "error": "Claude response parsing failed",
            "details": str(e),
            "raw": raw_text
        }


# -----------------------------
# AGENT PIPELINE
# -----------------------------
def agent(columns):
    mapped, unmapped = map_known(columns)

    claude_analysis = analyze_with_claude(unmapped)

    return {
        "mapped_columns": mapped,
        "unmapped_columns": unmapped,
        "claude_analysis": claude_analysis
    }


# -----------------------------
# API ROUTES
# -----------------------------
@app.get("/")
def root():
    return {"message": "FinMap AI + Claude Schema Assistant Running"}


@app.post("/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))

    result = agent(list(df.columns))

    return {
        "columns_detected": list(df.columns),
        "result": result
    }