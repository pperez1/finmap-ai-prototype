from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import os
import json
import re
from dotenv import load_dotenv
from anthropic import Anthropic
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
AUTO_MAP_THRESHOLD = 0.9  # Phase 3 guardrail

MEMORY_FILE = "schema_memory.json"
LOG_FILE = "events.log"

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not API_KEY:
    raise ValueError("Missing ANTHROPIC_API_KEY")

client = Anthropic(api_key=API_KEY)

# -----------------------------
# FASTAPI
# -----------------------------
app = FastAPI(title="FinMap AI - Guardrailed Schema Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# BASE SCHEMA
# -----------------------------
KNOWN_SCHEMA = {
    "Revenue": "revenue",
    "OperatingCost": "operating_expense",
    "DSCR": "dscr"
}

PROTECTED_FIELDS = set(KNOWN_SCHEMA.values())

# -----------------------------
# MEMORY
# -----------------------------
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

# -----------------------------
# LOGGING
# -----------------------------
def log_event(event):
    event["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

# -----------------------------
# JSON CLEANER
# -----------------------------
def extract_json(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

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

    columns_str = "\n".join(f"- {c}" for c in columns)

    prompt = f"""
You are analyzing UNKNOWN column names from a financial dataset.

Rules:
- Do NOT assume they map to known schema
- Do NOT force equivalence (e.g., revenue != income)
- Provide cautious interpretations

For each column return:
- possible_meaning
- confidence (0 to 1)
- reasoning

Columns:
{columns_str}

Return ONLY raw JSON.
"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text

    try:
        cleaned = extract_json(raw)
        return json.loads(cleaned)
    except:
        return {"error": "parse_failed", "raw": raw}

# -----------------------------
# AGENT PIPELINE
# -----------------------------
def agent(columns):
    memory = load_memory()

    mapped = {}
    unmapped = []
    ai_suggestions = {}

    # 1. Known + Learned mapping
    for col in columns:
        if col in KNOWN_SCHEMA:
            mapped[col] = {
                "mapping": KNOWN_SCHEMA[col],
                "confidence": 1.0,
                "source": "base_schema"
            }

        elif col in memory:
            mapped[col] = {
                "mapping": memory[col],
                "confidence": 1.0,
                "source": "learned"
            }

        else:
            unmapped.append(col)

    # 2. AI interpretation
    ai_analysis = analyze_with_claude(unmapped)

    # 3. Apply guardrails (auto-map if high confidence)
    for col, data in ai_analysis.items():
        confidence = data.get("confidence", 0)

        if confidence >= AUTO_MAP_THRESHOLD:
            proposed = data.get("possible_meaning", "unknown").lower().replace(" ", "_")

            # Guardrail: do not overwrite protected schema
            if proposed not in PROTECTED_FIELDS:
                mapped[col] = {
                    "mapping": proposed,
                    "confidence": confidence,
                    "source": "ai_auto"
                }

                log_event({
                    "event": "auto_mapped",
                    "column": col,
                    "mapping": proposed,
                    "confidence": confidence
                })
            else:
                ai_suggestions[col] = data

        else:
            ai_suggestions[col] = data

    # 4. Log run
    log_event({
        "event": "mapping_run",
        "columns": columns,
        "mapped": list(mapped.keys()),
        "unmapped": list(ai_suggestions.keys())
    })

    return {
        "mapped_columns": mapped,
        "unmapped_columns": list(ai_suggestions.keys()),
        "ai_suggestions": ai_suggestions
    }

# -----------------------------
# API ROUTES
# -----------------------------
@app.get("/")
def root():
    return {"message": "FinMap AI Running with Guardrails"}

@app.post("/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))

    result = agent(list(df.columns))

    return {
        "columns_detected": list(df.columns),
        "result": result
    }

# -----------------------------
# USER CONFIRMATION
# -----------------------------
@app.post("/confirm-mapping")
def confirm_mapping(mapping: dict):
    memory = load_memory()

    for col, mapped_value in mapping.items():
        memory[col] = mapped_value

        log_event({
            "event": "mapping_confirmed",
            "column": col,
            "mapped_to": mapped_value
        })

    save_memory(memory)

    return {"status": "saved", "memory": memory}