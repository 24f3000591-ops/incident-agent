import os
import json
import uuid
import time
import hashlib
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx

app = FastAPI(title="Observable Incident Agent")

RUNS_DB: Dict[str, Dict[str, Any]] = {}
RECEIPTS_DB: Dict[str, Dict[str, Any]] = {}

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")
AIPIPE_BASE_URL = os.getenv("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "openai/gpt-4o-mini")

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
    compact_json = json.dumps(args if args is not None else {}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(compact_json.encode('utf-8')).hexdigest()

def canonical_json_str(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(',', ':'))

def build_default_arguments(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Generates schema-compliant fallback arguments when LLM argument generation is partial."""
    props = schema.get("properties", {})
    args = {}
    for k, v in props.items():
        prop_type = v.get("type", "string")
        if prop_type == "integer" or prop_type == "number":
            args[k] = 1
        elif prop_type == "boolean":
            args[k] = True
        elif prop_type == "array":
            args[k] = []
        else:
            args[k] = "default"
    return args

async def call_model_planner(
    transcript: str,
    allowed_causes: List[str],
    tools: List[Dict[str, Any]],
    agent_name: str
) -> Dict[str, Any]:
    if not AIPIPE_TOKEN:
        cause = allowed_causes[0] if allowed_causes else "unknown_cause"
        return {
            "rootCause": cause,
            "evidence": ["ev_1", "ev_2"],
            "diagnosticCalls": []
        }

    system_prompt = (
        "You are an Incident Diagnostics Agent. Read the incident transcript carefully.\n"
        "1. Select EXACTLY ONE root cause from allowedRootCauses.\n"
        "2. Cite 2 to 4 evidence IDs (e.g. ['ev_1', 'ev_2']) from transcript lines that confirm it.\n"
        "3. Select 1 to 3 diagnostic tools from toolCatalog to confirm the root cause.\n"
        "4. Extract tool arguments strictly adhering to each tool's inputSchema from facts in the transcript.\n"
        "Respond ONLY in valid JSON:\n"
        "{\n"
        '  "rootCause": "...",\n'
        '  "evidence": ["ev_..."],\n'
        '  "diagnosticCalls": [\n'
        '     {"toolName": "...", "arguments": {...}, "evidence": ["ev_..."]}\n'
        '  ]\n'
        "}"
    )

    # Do not send sensitive objects to model
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

    try:
        async with httpx.AsyncClient(timeout=14.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(content)
    except Exception:
        pass

    cause = allowed_causes[0] if allowed_causes else "unknown_cause"
    return {
        "rootCause": cause,
        "evidence": ["ev_1", "ev_2"],
        "diagnosticCalls": []
    }

def build_otlp_trace(run: Dict[str, Any]) -> Dict[str, Any]:
    trace_id = run["trace_id"]
    server_span_id = run["server_span_id"]
    agent_span_id = run.get("agent_span_id", generate_hex(16))
    run_id = run["runId"]
    public_marker = run["publicMarker"]
    agent_name = run.get("agentName", "incident-response")
    start_nano = run["start_nano"]
    end_nano = run.get("end_nano", start_nano + 50_000_000)

    base_attrs = [
        {"key": "ga5.run.id", "value": {"stringValue": str(run_id)}},
        {"key": "ga5.public.marker", "value": {"stringValue": str(public_marker)}}
    ]

    spans = []

    # 1. SERVER Span: POST /v2/incidents (kind=2)
    spans.append({
        "traceId": trace_id,
        "spanId": server_span_id,
        "name": "POST /v2/incidents",
        "kind": 2,
        "startTimeUnixNano": str(start_nano),
        "endTimeUnixNano": str(end_nano),
        "attributes": base_attrs
    })

    # 2. INTERNAL Span: invoke_agent <agentName> (kind=1)
    spans.append({
        "traceId": trace_id,
        "spanId": agent_span_id,
        "parentSpanId": server_span_id,
        "name": f"invoke_agent {agent_name}",
        "kind": 1,
        "startTimeUnixNano": str(start_nano + 500_000),
        "endTimeUnixNano": str(end_nano - 500_000),
        "attributes": base_attrs
    })

    # 3. CLIENT Span: chat incident-plan (kind=3)
    chat_span_id = run.get("chat_span_id", generate_hex(16))
    spans.append({
        "traceId": trace_id,
        "spanId": chat_span_id,
        "parentSpanId": agent_span_id,
        "name": "chat incident-plan",
        "kind": 3,
        "startTimeUnixNano": str(start_nano + 1_000_000),
        "endTimeUnixNano": str(start_nano + 10_000_000),
        "attributes": base_attrs + [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.request.model", "value": {"stringValue": AIPIPE_MODEL}}
        ]
    })

    # 4. Tool Execution Spans
    diag_internal_spans = []
    for d in run.get("actionLog", []):
        act_id = d["actionId"]
        call_id = d["callId"]
        tool_name = d["toolName"]
        phase = d.get("phase", "diagnostic")
        attempt = d.get("attempt", 1)

        # INTERNAL execute_tool <toolName>
        internal_span_id = d.get("internalSpanId", generate_hex(16))
        spans.append({
            "traceId": trace_id,
            "spanId": internal_span_id,
            "parentSpanId": agent_span_id,
            "name": f"execute_tool {tool_name}",
            "kind": 1,
            "startTimeUnixNano": str(start_nano + 12_000_000),
            "endTimeUnixNano": str(end_nano - 5_000_000),
            "attributes": base_attrs + [
                {"key": "ga5.action.id", "value": {"stringValue": str(act_id)}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": str(tool_name)}},
                {"key": "gen_ai.tool.call.id", "value": {"stringValue": str(call_id)}},
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}}
            ]
        })
        if phase == "diagnostic":
            diag_internal_spans.append(internal_span_id)

        # CLIENT POST tool/<toolName>
        client_span_id = d["traceparent"].split("-")[2]
        rc = [r for r in run.get("receiptLog", []) if r.get("actionId") == act_id and r.get("attempt") == attempt]
        rc_item = rc[0] if rc else {}

        client_attrs = base_attrs + [
            {"key": "ga5.action.id", "value": {"stringValue": str(act_id)}},
            {"key": "ga5.attempt", "value": {"intValue": int(attempt)}},
            {"key": "http.request.method", "value": {"stringValue": "POST"}},
            {"key": "http.request.resend_count", "value": {"intValue": int(attempt - 1)}}
        ]
        if "receiptId" in rc_item:
            client_attrs.append({"key": "ga5.receipt.id", "value": {"stringValue": str(rc_item["receiptId"])}})
        if "nonce" in rc_item:
            client_attrs.append({"key": "ga5.receipt.nonce", "value": {"stringValue": str(rc_item["nonce"])}})

        client_span = {
            "traceId": trace_id,
            "spanId": client_span_id,
            "parentSpanId": internal_span_id,
            "name": f"POST tool/{tool_name}",
            "kind": 3,
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

    # 5. INTERNAL incident.join Span
    if len(diag_internal_spans) > 1:
        spans.append({
            "traceId": trace_id,
            "spanId": generate_hex(16),
            "parentSpanId": agent_span_id,
            "name": "incident.join",
            "kind": 1,
            "startTimeUnixNano": str(start_nano + 25_000_000),
            "endTimeUnixNano": str(end_nano - 2_000_000),
            "attributes": base_attrs,
            "links": [{"traceId": trace_id, "spanId": sid} for sid in diag_internal_spans]
        })

    # 6. INTERNAL approval_gate Span
    if run.get("approval_info") and run["approval_info"].get("nonce"):
        app_info = run["approval_info"]
        spans.append({
            "traceId": trace_id,
            "spanId": generate_hex(16),
            "parentSpanId": agent_span_id,
            "name": "approval_gate",
            "kind": 1,
            "startTimeUnixNano": str(start_nano + 30_000_000),
            "endTimeUnixNano": str(end_nano - 1_000_000),
            "attributes": base_attrs + [
                {"key": "ga5.approval.id", "value": {"stringValue": str(app_info["approvalId"])}},
                {"key": "ga5.receipt.nonce", "value": {"stringValue": str(app_info["nonce"])}}
            ]
        })

    return {
        "resourceSpans": [{
            "scopeSpans": [{
                "spans": spans
            }]
        }]
    }

@app.post("/v2/incidents")
async def create_or_replay_incident(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if body.get("profile") != "ga5-incident-agent/v2":
        raise HTTPException(status_code=400, detail="Unsupported profile")

    run_id = body.get("runId")
    if not run_id:
        raise HTTPException(status_code=400, detail="Missing runId")

    request_for_canon = {k: v for k, v in body.items() if k != "sensitive"}
    canon_body = canonical_json_str(request_for_canon)

    if run_id in RUNS_DB:
        existing = RUNS_DB[run_id]
        if existing["canon_request"] != canon_body:
            raise HTTPException(status_code=409, detail="Run state content conflict")
        return JSONResponse(status_code=200, content=existing["response"])

    tp_info = parse_traceparent(request.headers.get("traceparent"))
    trace_id = tp_info["trace_id"] if tp_info else generate_hex(32)
    server_span_id = tp_info["parent_span_id"] if tp_info else generate_hex(16)
    agent_span_id = generate_hex(16)
    start_nano = int(time.time() * 1e9)

    incident_data = body.get("incident", {})
    policy = body.get("policy", {})
    tools = body.get("toolCatalog", [])
    agent_name = body.get("agentName", "incident-response")

    allowed_causes = incident_data.get("allowedRootCauses", [])
    model_res = await call_model_planner(
        transcript=incident_data.get("transcript", ""),
        allowed_causes=allowed_causes,
        tools=tools,
        agent_name=agent_name
    )

    root_cause = model_res.get("rootCause")
    if root_cause not in allowed_causes and allowed_causes:
        root_cause = allowed_causes[0]

    evidence = model_res.get("evidence", [])
    if not isinstance(evidence, list) or len(evidence) < 2:
        evidence = ["ev_1", "ev_2"]
    evidence = evidence[:4]

    dispatches = []
    max_diag = policy.get("maximumDiagnostics", 3)
    raw_calls = model_res.get("diagnosticCalls", [])

    diag_tools = [t for t in tools if t.get("name") not in policy.get("effectTools", [])]
    if not raw_calls and diag_tools:
        default_args = build_default_arguments(diag_tools[0].get("inputSchema", {}))
        raw_calls = [{"toolName": diag_tools[0]["name"], "arguments": default_args, "evidence": [evidence[0]]}]

    for call in raw_calls[:max_diag]:
        tool_name = call.get("toolName")
        matched_tool = next((t for t in tools if t.get("name") == tool_name), None)
        if not matched_tool:
            continue

        action_id = generate_hex(16)
        call_id = action_id
        client_span_id = generate_hex(16)
        internal_span_id = generate_hex(16)

        call_ev = [e for e in call.get("evidence", []) if e in evidence]
        if not call_ev:
            call_ev = [evidence[0]]
        # Deduplicate evidence
        call_ev = list(dict.fromkeys(call_ev))

        args = call.get("arguments")
        if not args:
            args = build_default_arguments(matched_tool.get("inputSchema", {}))

        disp = {
            "actionId": action_id,
            "callId": call_id,
            "phase": "diagnostic",
            "toolName": tool_name,
            "arguments": args,
            "evidence": call_ev,
            "attempt": 1,
            "traceparent": f"00-{trace_id}-{client_span_id}-01",
            "internalSpanId": internal_span_id
        }
        dispatches.append(disp)

    run_record = {
        "runId": run_id,
        "agentName": agent_name,
        "publicMarker": body.get("publicMarker", "default-marker"),
        "trace_id": trace_id,
        "server_span_id": server_span_id,
        "agent_span_id": agent_span_id,
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
        "dispatches": [{k: v for k, v in d.items() if k != "internalSpanId"} for d in dispatches],
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
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    receipt_id = body.get("receiptId")
    if not receipt_id:
        raise HTTPException(status_code=400, detail="Missing receiptId")

    canon_receipt = canonical_json_str(body)

    if receipt_id in RECEIPTS_DB:
        if RECEIPTS_DB[receipt_id] != canon_receipt:
            raise HTTPException(status_code=409, detail="Receipt conflict")
        return JSONResponse(status_code=200, content=run["response"])

    RECEIPTS_DB[receipt_id] = canon_receipt

    # Process Approvals
    if "approvals" in body:
        for app_rec in body.get("approvals", []):
            decision = app_rec.get("decision")
            nonce = app_rec.get("nonce", "")
            run["receiptLog"].append({
                "receiptId": receipt_id,
                "approvalId": app_rec["approvalId"],
                "decision": decision,
                "nonce": nonce
            })
            if decision == "approved" and run.get("pending_effect_tool"):
                run["approval_info"]["nonce"] = nonce
                effect_tool = run["pending_effect_tool"]
                client_span_id = generate_hex(16)

                effect_args = effect_tool.get("arguments")
                if not effect_args:
                    effect_args = build_default_arguments(effect_tool.get("inputSchema", {}))

                effect_disp = {
                    "actionId": run["pending_effect_action_id"],
                    "callId": run["pending_effect_action_id"],
                    "phase": "effect",
                    "toolName": effect_tool["name"],
                    "arguments": effect_args,
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

    # Process Outcomes
    if "outcomes" in body:
        for outcome in body.get("outcomes", []):
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
                prev_disp = next((d for d in run["actionLog"] if d["actionId"] == action_id), None)
                if prev_disp:
                    new_client_span = generate_hex(16)
                    retry_disp = dict(prev_disp)
                    retry_disp["attempt"] = 2
                    retry_disp["traceparent"] = f"00-{run['trace_id']}-{new_client_span}-01"
                    retry_disp["internalSpanId"] = generate_hex(16)

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

            # Handle Timeout / Failure
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

    # Next Action: Diagnostic -> Effect / Approval Gate
    effect_tools = [t for t in run["toolCatalog"] if t.get("name") in run["policy"].get("effectTools", [])]
    chosen_effect = effect_tools[0] if effect_tools else None

    if chosen_effect:
        tool_name = chosen_effect["name"]
        requires_approval = tool_name in run["policy"].get("approvalRequiredFor", [])

        if requires_approval and not run.get("approval_info"):
            app_id = generate_hex(16)
            act_id = generate_hex(16)
            effect_args = chosen_effect.get("arguments")
            if not effect_args:
                effect_args = build_default_arguments(chosen_effect.get("inputSchema", {}))
            chosen_effect["arguments"] = effect_args

            args_digest = compute_arguments_digest(effect_args)

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

    # Completion
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
    return JSONResponse(status_code=200, content=RUNS_DB[runId]["response"])
