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
AUTO_MAP_THRESHOLD = 0.9

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
app = FastAPI(title="FinMap AI - Explainable Schema Engine")

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
# CLAUDE: INTERPRETATION
# -----------------------------
def interpret_columns(columns):
    if not columns:
        return {}

    prompt = f"""
You are analyzing UNKNOWN column names from a financial dataset.

Rules:
- Do NOT assume mapping to known schema
- Do NOT force equivalence (e.g., revenue != income)
- Be cautious and explain uncertainty

For each column return:
- possible_meaning
- confidence (0 to 1)
- reasoning

Columns:
{columns}

Return ONLY valid JSON.
"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text

    try:
        return json.loads(extract_json(raw))
    except:
        return {"_error": raw}

# -----------------------------
# CLAUDE: SCHEMA NAMING
# -----------------------------
def generate_schema_names(interpretation):
    if not interpretation:
        return {}

    prompt = f"""
You are generating database column names.

Rules:
- snake_case only
- max 3 words
- concise and standardized
- different meanings MUST have different names
- do NOT repeat schema names

Input:
{json.dumps(interpretation, indent=2)}

Return JSON:
{{ "ColumnName": "schema_name" }}
"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text

    try:
        return json.loads(extract_json(raw))
    except:
        return {}

# -----------------------------
# AGENT PIPELINE
# -----------------------------
def run_agent(columns):
    memory = load_memory()

    mapped = {}
    unmapped = []

    # 1. Known + Learned (NO TOKENS)
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

    # 2. Short circuit (no AI needed)
    if not unmapped:
        return {
            "mapped_columns": mapped,
            "unmapped_columns": [],
            "ai_suggestions": {}
        }

    # 3. AI only on unknown
    interpretation = interpret_columns(unmapped)
    schema_names = generate_schema_names(interpretation)

    ai_suggestions = {}

    # 4. Combine with explainability
    for col in unmapped:
        data = interpretation.get(col)

        # fallback
        if not data:
            match = next(
                (v for k, v in interpretation.items() if k.lower() == col.lower()),
                None
            )
            data = match

        if not isinstance(data, dict):
            data = {
                "possible_meaning": None,
                "confidence": 0,
                "reasoning": "AI parsing failed"
            }

        suggested_schema = schema_names.get(col)
        confidence = data.get("confidence", 0)

        # ✅ AUTO MAP WITH EXPLANATION INCLUDED
        if (
            confidence >= AUTO_MAP_THRESHOLD
            and suggested_schema
            and suggested_schema not in PROTECTED_FIELDS
        ):
            mapped[col] = {
                "mapping": suggested_schema,
                "confidence": confidence,
                "source": "ai_auto",
                "possible_meaning": data.get("possible_meaning"),
                "reasoning": data.get("reasoning")
            }

            log_event({
                "event": "auto_mapped",
                "column": col,
                "mapping": suggested_schema,
                "confidence": confidence
            })

        else:
            ai_suggestions[col] = {
                "possible_meaning": data.get("possible_meaning"),
                "confidence": confidence,
                "reasoning": data.get("reasoning"),
                "suggested_schema": suggested_schema
            }

    # log run
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
# ROUTES
# -----------------------------
@app.get("/")
def root():
    return {"message": "FinMap AI Running (Explainable Mode)"}


@app.post("/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))

    result = run_agent(list(df.columns))

    return {
        "columns_detected": list(df.columns),
        "result": result
    }


@app.post("/confirm-mapping")
def confirm_mapping(mapping: dict):
    memory = load_memory()

    for col, val in mapping.items():
        memory[col] = val

        log_event({
            "event": "mapping_confirmed",
            "column": col,
            "mapped_to": val
        })

    save_memory(memory)

    return {"status": "saved", "memory": memory}


@app.post("/reset-system")
def reset_system():
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    return {"status": "reset_complete"}