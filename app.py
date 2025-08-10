#!/usr/bin/env python3
"""
Bridge Ollama <-> Zabbix MCP Server

Prereqs:
  pip install "mcp[cli]" ollama

Env vars you may set:
  MCP_URL      (default: http://localhost:8820/mcp ; try http://localhost:8820 if your server mounts MCP at /)
  OLLAMA_HOST  (default: http://localhost:11434)
  OLLAMA_MODEL (default: qwen2.5:3b-instruct or any tool-capable model you pulled)
"""

import os
import json
import asyncio
from typing import Any, Dict

import ollama
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.context import RequestContext
from mcp import types as mtypes


# ---------- Settings ----------
MCP_URL = os.getenv("MCP_URL", "http://localhost:8820/mcp")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")  # or llama3.1:8b-instruct, phi3:mini, etc.

# A small system message to make the model pick a tool deterministically.
SYSTEM_PROMPT = """You are a tool-using assistant for Zabbix.
You can call exactly ONE tool by returning strict JSON:
{"tool": "<tool_name>", "arguments": {...}}  and nothing else.

Only use tools from the provided list. If the user asks a general question,
prefer read-only tools like problem_get, host_get, history_get, trend_get.
If unsure, ask for clarification by returning:
{"tool": null, "arguments": {}}."""

# Optional few-shot examples help the model pick correct tools/args
FEWSHOTS = [
    {
        "user": "List recent critical problems (top 5)",
        "json": {"tool": "problem_get", "arguments": {"recent": True, "severity": ["5"], "limit": 5}},
    },
    {
        "user": "Get hosts in Linux servers group",
        "json": {"tool": "host_get", "arguments": {"groupids": ["1"]}},
    },
]


# ---------- MCP helper ----------
async def with_mcp_session(mcp_url: str):
    """
    Async context manager that yields an MCP ClientSession over Streamable HTTP transport.
    """
    async with streamablehttp_client(mcp_url) as (read, write, _close):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def list_tools(session: ClientSession) -> Dict[str, mtypes.Tool]:
    tools_resp = await session.list_tools()
    return {t.name: t for t in tools_resp.tools}


async def call_tool(session: ClientSession, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    result = await session.call_tool(name, arguments=arguments or {})
    # Prefer structured content when present (spec 2025-06-18); otherwise fall back to text blocks
    if result.structuredContent is not None:
        return {"structured": result.structuredContent}
    # Fallback: concatenate any text content
    joined = []
    for c in result.content or []:
        if isinstance(c, mtypes.TextContent):
            joined.append(c.text)
    return {"text": "\n".join(joined) if joined else None}


# ---------- Ollama helper ----------
def choose_tool_with_ollama(user_prompt: str, tools_index: Dict[str, mtypes.Tool]) -> Dict[str, Any]:
    """
    Ask Ollama to choose a tool and arguments by returning STRICT JSON.
    We pass the available tool names + parameter schemas in a compact way.
    """
    # Compress tool schemas: just names + params (if present)
    tool_summaries = []
    for name, t in tools_index.items():
        params = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)  # SDK compat
        tool_summaries.append({"name": name, "schema": params})

    # Build the conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "Available tools (name + JSON schema): " + json.dumps(tool_summaries)[:25000]},
    ]
    for ex in FEWSHOTS:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": json.dumps(ex["json"])})
    messages.append({"role": "user", "content": user_prompt})

    # Call Ollama
    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.chat(model=OLLAMA_MODEL, messages=messages)
    text = resp["message"]["content"].strip()

    # Try to parse strict JSON; if it fails, return null-tool
    try:
        parsed = json.loads(text)
        # basic shape check
        if not isinstance(parsed, dict) or "tool" not in parsed or "arguments" not in parsed:
            raise ValueError("Bad shape")
        return parsed
    except Exception:
        return {"tool": None, "arguments": {}}


# ---------- End-to-end demo ----------
async def main():
    print(f"[+] Connecting to MCP server at {MCP_URL}")
    async for session in with_mcp_session(MCP_URL):
        tools_index = await list_tools(session)
        print(f"[+] MCP tools discovered: {', '.join(sorted(tools_index.keys()))}")

        # --- Example queries you can change ---
        demo_queries = [
            "Show the 5 most recent problems",
            "List all hosts in Zabbix",
            "Get Zabbix API version",
        ]

        for q in demo_queries:
            print(f"\n=== USER: {q}")
            choice = choose_tool_with_ollama(q, tools_index)
            print(f"[model->tool] {choice}")

            name = choice.get("tool")
            args = choice.get("arguments", {})

            if not name:
                print("No tool selected by the model. You can refine the prompt.")
                continue
            if name not in tools_index:
                print(f"Model selected unknown tool: {name}")
                continue

            print(f"[+] Calling MCP tool: {name} with args: {args}")
            result = await call_tool(session, name, args)
            print("[RESULT]", json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

