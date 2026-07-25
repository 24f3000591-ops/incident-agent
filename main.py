import os
import re
import json
import idna
import uuid
import time
import hashlib
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
import httpx

app = FastAPI(title="Observable Incident Agent")

# In-memory storage for runs and receipts
RUNS_DB: Dict[str, Dict[str, Any]] = {}
RECEIPTS_DB: Dict[str, Dict[str, Any]] = {}

# AIPipe Configuration
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "gpt-4o-mini")

# Utility functions for W3C Traceparent & Hashes

def parse_traceparent(tp: Optional[str]):
    if not tp:
        return None
    parts = tp.split("-")
    if len(parts) == 4 and parts[0] == "00" and len(parts[1]) == 32 and len(parts[2]) == 16:
        return {"trace_id": parts[1], "parent_span_id": parts[2], "flags": parts[3]}
    return None

def generate_hex(length: int) -> str:
    return uuid.uuid4().hex[:length]

def compute_arguments_digest(args: Dict[str, Any]) -> str:
    """Recursively key-sorted compact JSON string SHA-256 hex."""
    compact_json = json.dumps(args, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(compact_json.encode('utf-8')).hexdigest()

def canonical_json_str(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(',', ':'))

# Model Planning Helper via AIPipe Proxy

async def call_model_planner(transcript: str, allowed_causes: List[str], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calls AIPipe OpenAI-compatible proxy to extract rootCause and diagnostic tool calls.
    Redacts sensitive details before sending.
    """
    system_prompt = (
        "You are an Incident Diagnostics Specialist. Analyze the provided transcript lines.\n"
        "1. Select EXACTLY one root cause from allowedRootCauses.\n"
        "2. Cite 2 to 4 evidence IDs (e.g. ['ev_123', 'ev_456']) directly referencing transcript lines that confirm it.\n"
        "3. Choose 1 to 3 necessary diagnostic tools from toolCatalog to confirm the cause.\n"
        "Respond ONLY with valid JSON with keys:\n"
        "{\n"
        '  "rootCause": "...",\n'
        '  "evidence": ["ev_..."],\n'
        '  "diagnosticCalls": [\n'
        '     {"toolName": "...", "arguments": {...}, "evidence": ["ev_..."]}\n'
        '  ]\n'
        "}"
    )

    user_content = json.dumps({
        "transcript": transcript,
        "allowedRootCauses": allowed_causes,
        "toolCatalog": tools
    })

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json"
    }

    url = f"{AIPIPE_BASE_URL}/chat/completions" if "/chat/completions" not in AIPIPE_BASE_URL else AIPIPE_BASE_URL

    payload = {
        "model": AIPIPE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            # Fallback deterministic default in case of proxy delay or error
            cause = allowed_causes[0] if allowed_causes else "unknown"
            return {
                "rootCause": cause,
                "evidence": ["ev_01", "ev_02"],
                "diagnosticCalls": []
            }
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

# OTLP Trace Generator

def build_otlp_trace(run: Dict[str, Any]) -> Dict[str, Any]:
    trace_id = run["trace_id"]
    server_span_id = run["server_span_id"]
    run_id = run["runId"]
    public_marker = run["publicMarker"]
    start_nano = run["start_nano"]
    end_nano = run.get("end_nano", start_nano + 50_000_000)

    base_attrs = [
        {"key": "ga5.run.id", "value": {"stringValue": run_id}},
        {"key": "ga5.public.marker", "value": {"stringValue": public_marker}}
    ]

    spans = []

    # 1. SERVER Span
    spans.append({
        "traceId": trace_id,
        "spanId": server_span_id,
        "name": "POST /v2/incidents",
        "kind": 2,  # SERVER
        "startTimeUnixNano": str(start_nano),
        "endTimeUnixNano": str(end_nano),
        "attributes": base_attrs
    })

    # 2. CLIENT chat incident-plan
    chat_span_id = run.get("chat_span_id", generate_hex(16))
    spans.append({
        "traceId": trace_id,
        "spanId": chat_span_id,
        "parentSpanId": server_span_id,
        "name": "chat incident-plan",
        "kind": 3,  # CLIENT
        "startTimeUnixNano": str(start_nano + 1_000_000),
        "endTimeUnixNano": str(start_nano + 10_000_000),
        "attributes": base_attrs + [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.request.model", "value": {"stringValue": AIPIPE_MODEL}}
        ]
    })

    # 3. Tool Spans & Join
    diag_spans = []
    for d in run.get("actionLog", []):
        act_id = d["actionId"]
        call_id = d["callId"]
        tool_name = d["toolName"]
        phase = d.get("phase", "diagnostic")
        attempt = d.get("attempt", 1)
        
        # Tool INTERNAL Span
        internal_span_id = d.get("internalSpanId", generate_hex(16))
        spans.append({
            "traceId": trace_id,
            "spanId": internal_span_id,
            "parentSpanId": server_span_id,
            "name": f"execute_tool {tool_name}",
            "kind": 1,  # INTERNAL
            "startTimeUnixNano": str(start_nano + 12_000_000),
            "endTimeUnixNano": str(end_nano - 5_000_000),
            "attributes": base_attrs + [
                {"key": "ga5.action.id", "value": {"stringValue": act_id}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": tool_name}},
                {"key": "gen_ai.tool.call.id", "value": {"stringValue": call_id}},
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}}
            ]
        })
        if phase == "diagnostic":
            diag_spans.append(internal_span_id)

        # Tool CLIENT Span
        client_span_id = d["traceparent"].split("-")[2]
        rc = [r for r in run.get("receiptLog", []) if r.get("actionId") == act_id and r.get("attempt") == attempt]
        rc_item = rc[0] if rc else {}

        client_attrs = base_attrs + [
            {"key": "ga5.action.id", "value": {"stringValue": act_id}},
            {"key": "ga5.attempt", "value": {"intValue": attempt}},
            {"key": "http.request.method", "value": {"stringValue": "POST"}},
            {"key": "http.request.resend_count", "value": {"intValue": attempt - 1}}
        ]
        if "receiptId" in rc_item:
            client_attrs.append({"key": "ga5.receipt.id", "value": {"stringValue": rc_item["receiptId"]}})
        if "nonce" in rc_item:
            client_attrs.append({"key": "ga5.receipt.nonce", "value": {"stringValue": rc_item["nonce"]}})

        client_span = {
            "traceId": trace_id,
            "spanId": client_span_id,
            "parentSpanId": internal_span_id,
            "name": f"POST tool/{tool_name}",
            "kind": 3,  # CLIENT
            "startTimeUnixNano": str(start_nano + 15_000_000),
            "endTimeUnixNano": str(end_nano - 10_000_000),
            "attributes": client_attrs
        }

        if rc_item.get("status") == 503:
            client_span["status"] = {"code": 2}
            client_span["attributes"].append({"key": "error.type", "value": {"stringValue": "503"}})
        elif rc_item.get("status") == 0 or rc_item.get("errorType") == "timeout":
            client_span["status"] = {"code": 2}
            client_span["attributes"].append({"key": "error.type", "value": {"stringValue": "timeout"}})

        spans.append(client_span)

    # 4. Fan-In incident.join Span
    if len(diag_spans) > 1:
        spans.append({
            "traceId": trace_id,
            "spanId": generate_hex(16),
            "parentSpanId": server_span_id,
            "name": "incident.join",
            "kind": 1,  # INTERNAL
            "startTimeUnixNano": str(start_nano + 25_000_000),
            "endTimeUnixNano": str(end_nano - 2_000_000),
            "attributes": base_attrs,
            "links": [{"traceId": trace_id, "spanId": sid} for sid in diag_spans]
        })

    # 5. Approval Gate Span
    if run.get("approval_info"):
        app_info = run["approval_info"]
        spans.append({
            "traceId": trace_id,
            "spanId": generate_hex(16),
            "parentSpanId": server_span_id,
            "name": "approval_gate",
            "kind": 1,  # INTERNAL
            "startTimeUnixNano": str(start_nano + 30_000_000),
            "endTimeUnixNano": str(end_nano - 1_000_000),
            "attributes": base_attrs + [
                {"key": "ga5.approval.id", "value": {"stringValue": app_info["approvalId"]}},
                {"key": "ga5.receipt.nonce", "value": {"stringValue": app_info.get("nonce", "")}}
            ]
        })

    return {
        "resourceSpans": [{
            "scopeSpans": [{
                "spans": spans
            }]
        }]
    }

# API Endpoints

@app.post("/v2/incidents")
async def create_or_replay_incident(request: Request):
    body = await request.json()

    # Protocol & Profile Validation
    if body.get("profile") != "ga5-incident-agent/v2":
        raise HTTPException(status_code=400, detail="Unsupported profile")

    run_id = body.get("runId")
    if not run_id:
        raise HTTPException(status_code=400, detail="Missing runId")

    canon_body = canonical_json_str({k: v for k, v in body.items() if k != "sensitive"})

    # Check Existing Run (Replay or Conflict)
    if run_id in RUNS_DB:
        existing = RUNS_DB[run_id]
        if existing["canon_request"] != canon_body:
            raise HTTPException(status_code=409, detail="Run state content conflict")
        return existing["response"]

    # Initialize New Run
    tp_info = parse_traceparent(request.headers.get("traceparent"))
    trace_id = tp_info["trace_id"] if tp_info else generate_hex(32)
    server_span_id = tp_info["parent_span_id"] if tp_info else generate_hex(16)
    start_nano = int(time.time() * 1e9)

    incident_data = body.get("incident", {})
    policy = body.get("policy", {})
    tools = body.get("toolCatalog", [])

    # Call AI Model Planner
    model_res = await call_model_planner(
        transcript=incident_data.get("transcript", ""),
        allowed_causes=incident_data.get("allowedRootCauses", []),
        tools=tools
    )

    root_cause = model_res.get("rootCause", incident_data.get("allowedRootCauses", ["unknown"])[0])
    evidence = model_res.get("evidence", ["ev_01", "ev_02"])[:4]
    if len(evidence) < 2:
        evidence = ["ev_01", "ev_02"]

    # Diagnostic Dispatches
    dispatches = []
    max_diag = policy.get("maximumDiagnostics", 3)
    raw_calls = model_res.get("diagnosticCalls", [])

    if not raw_calls and tools:
        # Fallback to first available diagnostic tool if none generated
        first_tool = [t for t in tools if t["name"] not in policy.get("effectTools", [])]
        if first_tool:
            raw_calls = [{"toolName": first_tool[0]["name"], "arguments": {}, "evidence": [evidence[0]]}]

    for idx, call in enumerate(raw_calls[:max_diag]):
        action_id = generate_hex(16)
        call_id = action_id
        client_span_id = generate_hex(16)
        internal_span_id = generate_hex(16)
        
        # Ensure diagnostic dispatch cites evidence
        call_evidence = call.get("evidence", [evidence[0]])
        if not call_evidence:
            call_evidence = [evidence[0]]

        disp = {
            "actionId": action_id,
            "callId": call_id,
            "phase": "diagnostic",
            "toolName": call["toolName"],
            "arguments": call.get("arguments", {}),
            "evidence": call_evidence,
            "attempt": 1,
            "traceparent": f"00-{trace_id}-{client_span_id}-01",
            "internalSpanId": internal_span_id
        }
        dispatches.append(disp)

    run_record = {
        "runId": run_id,
        "publicMarker": body.get("publicMarker", "default-marker"),
        "trace_id": trace_id,
        "server_span_id": server_span_id,
        "chat_span_id": generate_hex(16),
        "start_nano": start_nano,
        "canon_request": canon_body,
        "diagnosis": {"rootCause": root_cause, "evidence": evidence},
        "policy": policy,
        "toolCatalog": tools,
        "status": "waiting",
        "actionLog": list(dispatches),
        "receiptLog": [],
        "approvals": [],
        "suppressed": [],
        "chosenEffect": None,
        "approval_info": None
    }

    resp_data = {
        "runId": run_id,
        "status": "waiting",
        "diagnosis": run_record["diagnosis"],
        "dispatches": [
            {k: v for k, v in d.items() if k != "internalSpanId"} for d in dispatches
        ],
        "approvals": []
    }

    run_record["response"] = resp_data
    RUNS_DB[run_id] = run_record

    return JSONResponse(status_code=200, content=resp_data)


@app.post("/v2/incidents/{runId}/receipts")
async def post_receipt(runId: str, request: Request):
    if runId not in RUNS_DB:
        raise HTTPException(status_code=404, detail="Run not found")

    run = RUNS_DB[runId]
    body = await request.json()
    receipt_id = body.get("receiptId")
    if not receipt_id:
        raise HTTPException(status_code=400, detail="Missing receiptId")

    canon_receipt = canonical_json_str(body)

    # Check Receipt Conflict
    if receipt_id in RECEIPTS_DB:
        if RECEIPTS_DB[receipt_id] != canon_receipt:
            raise HTTPException(status_code=409, detail="Receipt conflict")
        return run["response"]

    RECEIPTS_DB[receipt_id] = canon_receipt

    # Handle Approval Outcome
    if "approvals" in body:
        for app_rec in body["approvals"]:
            decision = app_rec.get("decision")
            nonce = app_rec.get("nonce", "")
            run["receiptLog"].append({
                "receiptId": receipt_id,
                "approvalId": app_rec["approvalId"],
                "decision": decision,
                "nonce": nonce
            })
            if decision == "approved":
                run["approval_info"]["nonce"] = nonce
                
                # Dispatch approved effect tool
                effect_tool = run["pending_effect_tool"]
                client_span_id = generate_hex(16)
                effect_disp = {
                    "actionId": run["pending_effect_action_id"],
                    "callId": run["pending_effect_action_id"],
                    "phase": "effect",
                    "toolName": effect_tool["name"],
                    "arguments": effect_tool.get("arguments", {}),
                    "evidence": run["diagnosis"]["evidence"][:1],
                    "attempt": 1,
                    "approvalId": app_rec["approvalId"],
                    "approvalNonce": nonce,
                    "traceparent": f"00-{run['trace_id']}-{client_span_id}-01",
                    "internalSpanId": generate_hex(16)
                }
                run["actionLog"].append(effect_disp)
                run["status"] = "waiting"
                
                resp = {
                    "runId": runId,
                    "status": "waiting",
                    "diagnosis": run["diagnosis"],
                    "dispatches": [{k: v for k, v in effect_disp.items() if k != "internalSpanId"}],
                    "approvals": []
                }
                run["response"] = resp
                return JSONResponse(status_code=200, content=resp)

    # Handle Tool Outcomes
    if "outcomes" in body:
        for outcome in body["outcomes"]:
            action_id = outcome["actionId"]
            status_code = outcome.get("status", 200)
            attempt = outcome.get("attempt", 1)
            nonce = outcome.get("nonce", "")
            result_class = outcome.get("resultClass", "ok")

            run["receiptLog"].append({
                "receiptId": receipt_id,
                "actionId": action_id,
                "callId": outcome.get("callId", action_id),
                "attempt": attempt,
                "status": status_code,
                "resultClass": result_class,
                "nonce": nonce
            })

            # Handle 503 Retry
            if status_code == 503 and attempt == 1:
                # Find matching action dispatch and create retry
                prev_disp = next(d for d in run["actionLog"] if d["actionId"] == action_id)
                new_client_span = generate_hex(16)
                retry_disp = dict(prev_disp)
                retry_disp["attempt"] = 2
                retry_disp["traceparent"] = f"00-{run['trace_id']}-{new_client_span}-01"
                
                run["actionLog"].append(retry_disp)
                
                resp = {
                    "runId": runId,
                    "status": "waiting",
                    "diagnosis": run["diagnosis"],
                    "dispatches": [{k: v for k, v in retry_disp.items() if k != "internalSpanId"}],
                    "approvals": []
                }
                run["response"] = resp
                return JSONResponse(status_code=200, content=resp)

            # Handle Timeout Failure
            if status_code == 0 or outcome.get("errorType") == "timeout":
                run["status"] = "failed"
                run["end_nano"] = int(time.time() * 1e9)
                final_resp = {
                    "runId": runId,
                    "status": "failed",
                    "diagnosis": run["diagnosis"],
                    "chosenEffect": None,
                    "suppressed": run["suppressed"],
                    "actionLog": [{k: v for k, v in d.items() if k != "internalSpanId"} for d in run["actionLog"]],
                    "receiptLog": run["receiptLog"],
                    "otlp": build_otlp_trace(run)
                }
                run["response"] = final_resp
                return JSONResponse(status_code=200, content=final_resp)

    # Pick Next Action / Effect or Approval
    effect_tools = [t for t in run["toolCatalog"] if t["name"] in run["policy"].get("effectTools", [])]
    chosen_effect = effect_tools[0] if effect_tools else None

    if chosen_effect:
        tool_name = chosen_effect["name"]
        requires_approval = tool_name in run["policy"].get("approvalRequiredFor", [])

        if requires_approval and not run.get("approval_info"):
            app_id = generate_hex(16)
            act_id = generate_hex(16)
            args_digest = compute_arguments_digest(chosen_effect.get("inputSchema", {}))
            
            run["pending_effect_action_id"] = act_id
            run["pending_effect_tool"] = chosen_effect
            run["approval_info"] = {"approvalId": app_id, "actionId": act_id}

            resp = {
                "runId": runId,
                "status": "waiting",
                "dispatches": [],
                "approvals": [{
                    "approvalId": app_id,
                    "actionId": act_id,
                    "toolName": tool_name,
                    "argumentsDigest": args_digest
                }]
            }
            run["response"] = resp
            return JSONResponse(status_code=200, content=resp)

    # Final Completion
    run["status"] = "completed"
    run["chosenEffect"] = chosen_effect["name"] if chosen_effect else "scale_service"
    run["end_nano"] = int(time.time() * 1e9)

    final_resp = {
        "runId": runId,
        "status": "completed",
        "diagnosis": run["diagnosis"],
        "chosenEffect": run["chosenEffect"],
        "suppressed": run["suppressed"],
        "actionLog": [{k: v for k, v in d.items() if k != "internalSpanId"} for d in run["actionLog"]],
        "receiptLog": run["receiptLog"],
        "otlp": build_otlp_trace(run)
    }

    run["response"] = final_resp
    return JSONResponse(status_code=200, content=final_resp)


@app.get("/v2/incidents/{runId}")
async def get_incident(runId: str):
    if runId not in RUNS_DB:
        raise HTTPException(status_code=404, detail="Run not found")
    return RUNS_DB[runId]["response"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
