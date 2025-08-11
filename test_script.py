#!/usr/bin/env python3
# Fixed + hardcoded values for MCP_URL, Ollama host, and model

import re, json, asyncio
from typing import Any, Dict, Iterable, Tuple, Optional
import ollama

from mcp import ClientSession, types as mtypes
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client

# ------------------ Hardcoded config ------------------
MCP_URL_ENV = "http://localhost:8820/mcp"
OLLAMA_HOST = "http://localhost:11434"
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

# ------------------ Helpers ------------------
def _force_json(s: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def choose_tool_with_ollama(user_prompt: str, tools: Dict[str, mtypes.Tool]) -> Dict[str, Any]:
    tool_summaries = []
    for name, t in tools.items():
        schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
        tool_summaries.append({"name": name, "schema": schema})

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "Tools: " + json.dumps(tool_summaries)[:25000]},
    ]
    for u, a in FEWSHOTS:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": json.dumps(a, ensure_ascii=False)})

    msgs.append({"role": "user", "content": user_prompt})

    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.chat(model=OLLAMA_MODEL, messages=msgs, format="json",
                       options={"temperature": 0})
    text = (resp.get("message") or {}).get("content", "").strip()

    parsed = _force_json(text)
    if isinstance(parsed, dict) and "tool" in parsed and "arguments" in parsed:
        return parsed

    s = user_prompt.lower()
    if "version" in s or "api" in s:
        return {"tool": "apiinfo_version", "arguments": {}}
    if any(k in s for k in ("problem", "incident", "alert")):
        return {"tool": "problem_get", "arguments": {"recent": True, "limit": 5}}
    if "host" in s:
        return {"tool": "host_get", "arguments": {}}
    return {"tool": None, "arguments": {}}

async def run_workflow(session: ClientSession):
    tools_resp = await session.list_tools()
    tools = {t.name: t for t in tools_resp.tools}
    print("[+] MCP tools:", ", ".join(sorted(tools.keys())))

    demos = [
        "Show the 5 most recent problems",
        "List all hosts in Zabbix",
        "Get Zabbix API version",
        "Retrieve monitoring templates"
    ]
    for q in demos:
        print(f"\n=== USER: {q}")
        choice = choose_tool_with_ollama(q, tools)
        print("[model->tool]", choice)
        name = choice.get("tool")
        args = choice.get("arguments", {})
        if not name or name not in tools:
            print("No/unknown tool chosen.")
            continue
        print(f"[+] Calling {name} with {args}")
        try:
            result = await session.call_tool(name, arguments=args)
            if getattr(result, "structuredContent", None) is not None:
                print("[RESULT]", json.dumps({"structured": result.structuredContent}, indent=2, ensure_ascii=False))
            else:
                texts = [c.text for c in (result.content or []) if hasattr(c, "text")]
                print("[RESULT]", "\n".join(texts) if texts else "(no text)")
        except Exception as e:
            print("[!] Tool call failed:", e)

async def connect_and_run(url_hint: str):
    candidates = [url_hint]
    if url_hint.endswith("/mcp"):
        candidates.append(url_hint.rsplit("/mcp", 1)[0])
    else:
        candidates.append(url_hint.rstrip("/") + "/mcp")

    last_err = None
    for base in candidates:
        try:
            async with streamablehttp_client(base) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    print(f"[+] MCP connected via STREAM at {base}")
                    await run_workflow(session)
                    return
        except Exception as e:
            last_err = e
            print(f"[i] STREAM failed at {base}: {e}")

        try:
            async with sse_client(base) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    print(f"[+] MCP connected via SSE at {base}")
                    await run_workflow(session)
                    return
        except Exception as e:
            last_err = e
            print(f"[i] SSE failed at {base}: {e}")

    raise RuntimeError(f"Could not connect to MCP at {url_hint}. Last error: {last_err}")

async def main():
    print(f"[+] Connecting to MCP server at {MCP_URL_ENV}")
    await connect_and_run(MCP_URL_ENV)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

