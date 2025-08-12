#!/usr/bin/env python3
import json, re, asyncio
from typing import Any, Dict, Iterable, Tuple, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import ollama
from mcp import ClientSession, types as mtypes
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client

# --- Hardcoded config (matches your working bridge) ---
MCP_URL = "http://zabbix-mcp:8820/mcp"
OLLAMA_HOST = "http://ollama:11434"
OLLAMA_MODEL = "qwen2.5:3b-instruct"

SYSTEM_PROMPT = (
    "You are a tool-using assistant for Zabbix.\n"
    'Return ONLY {"tool": "<name or null>", "arguments": {}} as strict JSON.\n'
    "Choose exactly one tool from the provided list (or null). No prose."
)

FEWSHOTS: Iterable[Tuple[str, Dict[str, Any]]] = (
    ("List recent critical problems (top 5)",
     {"tool": "problem_get", "arguments": {"recent": True, "severity": ["5"], "limit": 5}}),
    ("List all hosts",
     {"tool": "host_get", "arguments": {}}),
    ("API version",
     {"tool": "apiinfo_version", "arguments": {}}),
)

def _force_json(s: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _choose_tool_with_ollama(user_prompt: str, tools: Dict[str, mtypes.Tool]) -> Dict[str, Any]:
    tool_summaries = []
    for name, t in tools.items():
        schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
        tool_summaries.append({"name": name, "schema": schema})

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "Tools: " + json.dumps(tool_summaries)[:25000]},
    ]
    for u, a in FEWSHOTS:
        msgs += [{"role": "user", "content": u},
                 {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
    msgs.append({"role": "user", "content": user_prompt})

    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.chat(model=OLLAMA_MODEL, messages=msgs, format="json",
                       options={"temperature": 0})
    text = (resp.get("message") or {}).get("content", "").strip()

    parsed = _force_json(text)
    if isinstance(parsed, dict) and "tool" in parsed and "arguments" in parsed:
        return parsed

    # Heuristic fallback
    s = user_prompt.lower()
    if "version" in s or "api" in s: return {"tool": "apiinfo_version", "arguments": {}}
    if any(k in s for k in ("problem","incident","alert")):
        return {"tool": "problem_get", "arguments": {"recent": True, "limit": 5}}
    if "host" in s: return {"tool": "host_get", "arguments": {}}
    return {"tool": None, "arguments": {}}

async def _connect_session():
    # Try streamable then SSE, with /mcp already hardcoded in MCP_URL.
    last_err = None
    for client in (streamablehttp_client, sse_client):
        try:
            async with client(MCP_URL) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return session
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot connect to MCP at {MCP_URL}: {last_err}")

async def _run_query(user_prompt: str) -> Dict[str, Any]:
    async with await _connect_session() as session:  # type: ignore
        tools_resp = await session.list_tools()
        tools = {t.name: t for t in tools_resp.tools}
        choice = _choose_tool_with_ollama(user_prompt, tools)

        name = choice.get("tool")
        args = choice.get("arguments", {})
        if not name or name not in tools:
            return {"choice": choice, "result": None, "error": "no_tool_selected"}

        res = await session.call_tool(name, arguments=args)
        if getattr(res, "structuredContent", None) is not None:
            payload = {"structured": res.structuredContent}
        else:
            texts = [c.text for c in (res.content or []) if hasattr(c, "text")]
            payload = {"text": "\n".join(texts) if texts else None}

        return {"choice": choice, "result": payload, "error": None}

# --- FastAPI app ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    user_msg = (body.get("message") or "").strip()
    if not user_msg:
        return JSONResponse({"error": "empty_message"}, status_code=400)
    try:
        out = await _run_query(user_msg)
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

