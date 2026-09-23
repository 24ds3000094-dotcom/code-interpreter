import os
import sys
import json
import traceback
from io import StringIO
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI   # works with AI Pipe

load_dotenv()  # reads .env in the current working directory into os.environ

app = FastAPI()

# Enable CORS (required for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request / Response models ----------
class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    error: List[int]
    result: str

# ---------- Tool Function (exact as specified) ----------
def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact output.
    Returns:
        {
            "success": bool,
            "output": str  # Exact stdout or traceback
        }
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout

# ---------- AI Error Analysis (using AI Pipe + structured output) ----------
def analyze_error_with_ai(code: str, traceback_str: str) -> List[int]:
    """
    Use LLM with structured output to identify error line numbers.
    Uses AI Pipe (OpenAI-compatible endpoint).

    IMPORTANT: This is a best-effort enhancement. Any failure here
    (missing token, unsupported response_format, network error,
    malformed JSON, etc.) must NOT crash the main endpoint — we
    catch everything and fall back to an empty list.
    """
    token = os.environ.get("AIPIPE_TOKEN")
    if not token:
        print("[analyze_error_with_ai] AIPIPE_TOKEN not set, skipping AI analysis",
              file=sys.stderr)
        return []

    prompt = f"""
Analyze this Python code and its error traceback.
Identify the exact line number(s) where the error occurred.
Return ONLY the line numbers that caused the error.

CODE:
{code}

TRACEBACK:
{traceback_str}
"""

    try:
        client = OpenAI(
            base_url="https://aipipe.org/openrouter/v1",   # or /openai/v1
            api_key=token
        )

        response = client.chat.completions.create(
            model="openai/gpt-4.1-nano",   # confirmed working via AI Pipe/OpenRouter
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "error_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error_lines": {
                                "type": "array",
                                "items": {"type": "integer"}
                            }
                        },
                        "required": ["error_lines"],
                        "additionalProperties": False
                    }
                }
            }
        )

        content = response.choices[0].message.content
        result = json.loads(content)
        return result.get("error_lines", [])

    except Exception as e:
        # Log the real reason for debugging, but never let this
        # propagate up and 500 the endpoint.
        print(f"[analyze_error_with_ai] failed: {e}", file=sys.stderr)
        return []

# ---------- Main Endpoint ----------
@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest):
    # 1. Execute the code
    execution = execute_python_code(request.code)

    # 2. If successful -> return empty error list
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # 3. If error -> ask AI for line numbers (safe, never raises)
    try:
        error_lines = analyze_error_with_ai(request.code, execution["output"])
    except Exception as e:
        # Extra safety net in case anything above this still slips through
        print(f"[code_interpreter] unexpected failure calling AI analysis: {e}",
              file=sys.stderr)
        error_lines = []

    # 4. Return AI-detected lines + exact traceback
    return {
        "error": error_lines,
        "result": execution["output"]
    }

# Health check (optional but useful)
@app.get("/")
async def root():
    return {"status": "Code Interpreter is running"}