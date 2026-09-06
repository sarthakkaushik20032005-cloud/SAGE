#!/usr/bin/env python3
"""
SAGE — Complete System Integration Test Suite
=============================================
Tests that all components work together seamlessly without requiring live LLM weights:
  1. Module imports and prompt/static file verification
  2. Document processor integration (DOCX with embedded image, text, images)
  3. Robust Gemma JSON parser (handles markdown fences, <thought> tags, prose)
  4. Real Docker sandbox execution through CodeExecutionPipeline
  5. End-to-end multi-model sequential orchestration with mock inference:
     Gemma -> Document Analyzer (DOCX) -> Coder -> Docker Sandbox -> Gemma Synthesis
  6. End-to-end auto-repair loop:
     Gemma -> Coder (buggy code) -> Sandbox failure -> Auto-repair -> Sandbox success -> Gemma Synthesis
  7. FastAPI app and REST endpoints (/api/status, /api/stop, /, /api/chat validation)
"""
from __future__ import annotations

import io
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure SAGE directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from document_processor import DocumentProcessor
from code_executor.extractor import extract_code
from code_executor.sandbox import DockerSandbox
from code_executor.fixer import CodeFixer, MockCoder
from code_executor.pipeline import CodeExecutionPipeline
from orchestrator import Orchestrator
from model_client import ModelClient
from app import app


# ─── Test Runner ─────────────────────────────────────────────────────────────

class IntegrationTestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[tuple[str, str]] = []

    def run(self, name: str, fn):
        bar = "-" * max(0, 65 - len(name))
        print(f"\n[TEST] {name} {bar}")
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            print(f"       >>> PASSED ({elapsed:.2f}s)")
            self.passed += 1
        except Exception as e:
            elapsed = time.time() - t0
            import traceback
            tb = traceback.format_exc()
            print(f"       >>> FAILED ({elapsed:.2f}s): {e}")
            self.failed += 1
            self.errors.append((name, tb))

    def summary(self) -> int:
        print("\n" + "=" * 70)
        print(" SAGE SYSTEM INTEGRATION TEST SUMMARY")
        print("=" * 70)
        print(f" Total Tests : {self.passed + self.failed}")
        print(f" Passed      : {self.passed}  [OK]")
        print(f" Failed      : {self.failed}  {'[FAIL]' if self.failed else '[CLEAN]'}")
        if self.errors:
            print("\nFailures:")
            for name, tb in self.errors:
                print(f"  * {name}:\n{tb}")
        print("=" * 70)
        return 0 if self.failed == 0 else 1


runner = IntegrationTestRunner()


# ─── Test 1: Module Imports & Integrity ──────────────────────────────────────

def test_1_imports_and_assets():
    # Verify prompt files
    assert (config.PROMPTS_DIR / "agent_system.txt").exists(), "agent_system.txt missing"
    assert (config.PROMPTS_DIR / "coder_system.txt").exists(), "coder_system.txt missing"
    assert (config.PROMPTS_DIR / "document_system.txt").exists(), "document_system.txt missing"
    
    # Verify static files
    assert (config.STATIC_DIR / "index.html").exists(), "index.html missing"
    assert (config.STATIC_DIR / "app.js").exists(), "app.js missing"
    assert (config.STATIC_DIR / "style.css").exists(), "style.css missing"

    # Verify model configs
    assert "agent" in config.MODELS
    assert "coder" in config.MODELS
    assert "document_analyzer" in config.MODELS

    print("       Files, prompts, configs, and static UI assets present.")

runner.run("1. Imports & Configuration Assets", test_1_imports_and_assets)


# ─── Test 2: Document Processor Integration ──────────────────────────────────

def test_2_document_processor():
    # 1. Plain text processing
    mock_calls = []
    def mock_callback(prompt: str, meta: Optional[Dict[str, Any]]) -> str:
        mock_calls.append({"prompt": prompt, "meta": meta})
        if meta:
            return f"Parsed image ({meta['width']}x{meta['height']})"
        return "Processed text content successfully."

    # Test plain text chunking
    txt_path = config.TEMP_DIR / "integration_test.txt"
    txt_path.write_text("Revenue: $4.25M\nCustomers: 1840", encoding="utf-8")
    try:
        txt_res = DocumentProcessor.process(
            filepath=str(txt_path),
            task="Extract revenue",
            file_type="txt",
            call_model_fn=mock_callback
        )
        assert "Processed text content" in txt_res
    finally:
        txt_path.unlink(missing_ok=True)

    # 2. DOCX processing with embedded image & text
    docx_path = config.TEMP_DIR / "test_sample_report.docx"
    assert docx_path.exists(), "test_sample_report.docx missing from temp/"

    docx_calls = []
    def docx_callback(prompt: str, meta: Optional[Dict[str, Any]]) -> str:
        docx_calls.append({"prompt": prompt, "meta": meta})
        if meta:
            assert meta["data_uri"].startswith("data:image/")
            assert meta["width"] > 0 and meta["height"] > 0
            return "OCR: Total Revenue: $4,250,000 | Active Enterprise Customers: 1,840"
        return f"DOCX Summary: {prompt[:100]}"

    docx_res = DocumentProcessor.process(
        filepath=str(docx_path),
        task="Extract quarterly metrics",
        file_type="docx",
        call_model_fn=docx_callback
    )

    # Verify that image was extracted and sent to callback with metadata
    had_image = any(c["meta"] is not None for c in docx_calls)
    assert had_image, "DOCX did not extract embedded image"
    assert "DOCX Summary" in docx_res
    print(f"       DOCX processed successfully ({len(docx_calls)} model interactions, embedded image extracted).")

runner.run("2. Document Processor Integration (DOCX + OCR)", test_2_document_processor)


# ─── Test 3: Robust Gemma JSON Parser ────────────────────────────────────────

def test_3_json_parser():
    orch = Orchestrator()

    # Case A: Plain JSON
    a = orch._parse_gemma_json('{"type": "tool_calls", "calls": [{"tool": "coder"}]}')
    assert a is not None and a["type"] == "tool_calls"

    # Case B: JSON in markdown fences
    b = orch._parse_gemma_json('```json\n{"type": "final", "answer": "Done!"}\n```')
    assert b is not None and b["type"] == "final" and b["answer"] == "Done!"

    # Case C: JSON with <thought> tags (model reasoning)
    c = orch._parse_gemma_json(
        "<thought>\nLet's call the coder tool first.\n</thought>\n"
        "```json\n"
        '{"type": "tool_calls", "calls": [{"tool": "coder", "task": "test"}]}\n'
        "```"
    )
    assert c is not None and c["type"] == "tool_calls"
    assert c["calls"][0]["tool"] == "coder"

    # Case D: JSON with conversational prose preamble
    d = orch._parse_gemma_json(
        'Here is my response:\n{"type": "final", "answer": "All tasks complete."}\nHope this helps!'
    )
    assert d is not None and d["type"] == "final"
    assert d["answer"] == "All tasks complete."

    # Case E: Completely invalid JSON
    e = orch._parse_gemma_json("This has no JSON whatsoever.")
    assert e is None

    print("       JSON parser handles plain, fenced, reasoning <thought>, and prose wrappings.")

runner.run("3. Robust Gemma JSON Parser", test_3_json_parser)


# ─── Test 4: Real Docker Sandbox Execution via Pipeline ──────────────────────

def test_4_docker_sandbox_real():
    sandbox = DockerSandbox(config.SANDBOX)
    mock = MockCoder()
    pipeline = CodeExecutionPipeline(sandbox=sandbox, fixer=CodeFixer(mock), max_fix_attempts=1)

    code_payload = (
        "```python\n"
        "rev = 4250000\n"
        "cust = 1840\n"
        "avg = rev / cust\n"
        "print(f'Average revenue per customer: ${avg:.2f}')\n"
        "```"
    )

    result = pipeline.execute(task="Compute average revenue", llm_response=code_payload)

    assert result.succeeded, f"Sandbox failed: exit={result.exit_code} stderr={result.stderr}"
    assert "Average revenue per customer: $2309.78" in result.stdout
    assert result.exit_code == 0
    assert result.attempts == 1
    assert result.wall_time_ms > 0
    print(f"       Executed in Docker sandbox: {result.wall_time_ms:.0f}ms, output: {result.stdout.strip()}")

runner.run("4. Docker Sandbox Execution via Pipeline", test_4_docker_sandbox_real)


# ─── Test 5: End-to-End Multi-Model Sequential Orchestration ─────────────────

def test_5_orchestrator_e2e_mock():
    """
    Simulates the complete 3-model flow:
      Turn 1: Gemma requests document_analyzer tool
      DocumentProcessor extracts text & embedded image from DOCX -> Qwen3-VL analyzes it
      Turn 2: Gemma receives document analysis -> requests coder tool
      Qwen2.5-Coder writes Python code -> runs in real Docker sandbox
      Turn 3: Gemma receives sandbox stdout ($2309.78) -> generates final synthesis
    """
    orch = Orchestrator()
    docx_path = str(config.TEMP_DIR / "test_sample_report.docx")

    attachments_manifest = [
        {"ref": "file_1", "name": "test_sample_report.docx", "type": "docx", "size": os.path.getsize(docx_path)}
    ]
    file_map = {
        "file_1": {
            "ref": "file_1",
            "name": "test_sample_report.docx",
            "type": "docx",
            "size": os.path.getsize(docx_path),
            "path": docx_path
        }
    }

    # Sequence of mock responses for model_client.chat_completion
    call_step = [0]
    
    def mock_chat_completion(messages, temperature=None, max_tokens=None, **kwargs):
        step = call_step[0]
        call_step[0] += 1
        
        # Step 0: Gemma Loop 1 -> requests document_analyzer
        if step == 0:
            content = json.dumps({
                "type": "tool_calls",
                "calls": [
                    {
                        "tool": "document_analyzer",
                        "task": "Extract revenue and customer counts from report",
                        "input_refs": ["file_1"]
                    }
                ]
            })
        # Step 1: Qwen3-VL OCR on embedded image
        elif step == 1:
            content = "Extracted Image Data: Total Revenue: $4,250,000, Active Enterprise Customers: 1,840"
        # Step 2: Qwen3-VL document summary
        elif step == 2:
            content = "Report Summary: Total Revenue is $4,250,000 with 1,840 enterprise customers."
        # Step 3: Gemma Loop 2 -> requests coder
        elif step == 3:
            content = json.dumps({
                "type": "tool_calls",
                "calls": [
                    {
                        "tool": "coder",
                        "task": "Calculate average revenue per customer and display metric",
                        "context": "Revenue: 4250000, Customers: 1840"
                    }
                ]
            })
        # Step 4: Qwen2.5-Coder -> generates Python code
        elif step == 4:
            content = (
                "```python\n"
                "revenue = 4250000\n"
                "customers = 1840\n"
                "avg = revenue / customers\n"
                "print(f'CALCULATED_AVG={avg:.2f}')\n"
                "```"
            )
        # Step 5: Gemma Loop 3 -> final synthesis
        elif step == 5:
            content = json.dumps({
                "type": "final",
                "answer": (
                    "Based on the quarterly report, total revenue is $4,250,000 across 1,840 enterprise customers. "
                    "The computed average revenue per customer is $2,309.78 as verified via sandboxed execution."
                )
            })
        else:
            content = json.dumps({"type": "final", "answer": "Done"})

        return {
            "content": content,
            "duration": 0.05,
            "usage": {"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100},
            "timings": {},
            "raw": {}
        }

    with patch("orchestrator.model_client.chat_completion", side_effect=mock_chat_completion):
        with patch("orchestrator.model_manager.ensure_model", return_value=True):
            result = orch.run(
                user_objective="Analyze report and compute average revenue per customer",
                attachments_manifest=attachments_manifest,
                file_map=file_map
            )

    assert result["status"] == "success"
    assert "4,250,000" in result["answer"]
    assert "2,309.78" in result["answer"]
    assert result["telemetry"]["agent_calls"] == 3
    assert result["telemetry"]["document_calls"] >= 1
    assert result["telemetry"]["coder_calls"] == 1
    assert result["telemetry"]["sandbox_executions"] == 1
    
    # Verify trace timeline
    actors = [evt["actor"] for evt in result["trace"]]
    assert "gemma" in actors
    assert "document_analyzer" in actors
    assert "coder" in actors
    assert "sandbox" in actors

    # Verify sandbox trace captured real stdout
    sandbox_trace = next(evt for evt in result["trace"] if evt["actor"] == "sandbox")
    assert "CALCULATED_AVG=2309.78" in sandbox_trace["stdout_preview"]
    assert sandbox_trace["status"] == "success"

    print("       Full 3-model orchestration verified: Gemma -> DocAnalyzer -> Coder -> Sandbox -> Gemma.")

runner.run("5. End-to-End Orchestrator Pipeline (3 Models + Sandbox)", test_5_orchestrator_e2e_mock)


# ─── Test 6: Auto-Repair in Full Orchestrator Integration ────────────────────

def test_6_auto_repair_orchestrator():
    """
    Simulates:
      1. Gemma calls Coder.
      2. Coder initially generates code with a ZeroDivisionError.
      3. Sandbox detects failure (exit=1).
      4. Auto-repair loop invokes Coder repair -> returns corrected code.
      5. Sandbox re-executes fixed code -> success!
      6. Gemma receives attempts=2 and synthesizes answer.
    """
    orch = Orchestrator()
    step_counter = [0]

    def mock_chat(messages, **kwargs):
        s = step_counter[0]
        step_counter[0] += 1

        # Turn 0: Gemma requests coder
        if s == 0:
            return {
                "content": json.dumps({
                    "type": "tool_calls",
                    "calls": [{"tool": "coder", "task": "Compute ratio safely", "context": "denom=0"}]
                }),
                "duration": 0.05, "usage": {}
            }
        # Turn 1: Initial Coder output (buggy - division by zero)
        elif s == 1:
            return {
                "content": (
                    "```python\n"
                    "x = 10 / 0  # Buggy initial code\n"
                    "print(x)\n"
                    "```"
                ),
                "duration": 0.05, "usage": {}
            }
        # Turn 2: RealCoder repair output (fixed code)
        elif s == 2:
            return {
                "content": (
                    "```python\n"
                    "denom = 0\n"
                    "x = 10 / denom if denom != 0 else 0\n"
                    "print(f'SAFE_RESULT={x}')\n"
                    "```"
                ),
                "duration": 0.05, "usage": {}
            }
        # Turn 3: Gemma final synthesis
        elif s == 3:
            return {
                "content": json.dumps({
                    "type": "final",
                    "answer": "The calculation was safely performed with division-by-zero protection. Result: 0."
                }),
                "duration": 0.05, "usage": {}
            }
        return {"content": json.dumps({"type": "final", "answer": "ok"}), "duration": 0.05, "usage": {}}

    with patch("orchestrator.model_client.chat_completion", side_effect=mock_chat):
        with patch("orchestrator.model_manager.ensure_model", return_value=True):
            with patch("code_executor.fixer.RealCoder.generate") as mock_real_coder_gen:
                mock_real_coder_gen.return_value = (
                    "```python\n"
                    "denom = 0\n"
                    "x = 10 / denom if denom != 0 else 0\n"
                    "print(f'SAFE_RESULT={x}')\n"
                    "```"
                )
                res = orch.run(
                    user_objective="Compute ratio safely",
                    attachments_manifest=[],
                    file_map={}
                )

    assert res["status"] == "success"
    sandbox_trace = next(evt for evt in res["trace"] if evt["actor"] == "sandbox")
    assert sandbox_trace["attempts"] == 2
    assert "SAFE_RESULT=0" in sandbox_trace["stdout_preview"]
    assert "safely performed" in res["answer"]
    print("       Auto-repair loop verified: error caught -> repaired by coder -> sandbox verified -> Gemma synthesis.")

runner.run("6. Full Pipeline Code Auto-Repair Loop", test_6_auto_repair_orchestrator)


# ─── Test 7: FastAPI App Endpoints ───────────────────────────────────────────

def test_7_fastapi_endpoints():
    from fastapi.testclient import TestClient
    client = TestClient(app)

    # 1. /api/status
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "models" in data
    assert "agent" in data["models"]
    assert "coder" in data["models"]
    assert "document_analyzer" in data["models"]

    # 2. /api/stop
    res_stop = client.post("/api/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] == "stopped"

    # 3. GET / (serves HTML)
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "SAGE" in res_index.text

    # 4. POST /api/chat with empty prompt validation
    res_empty = client.post("/api/chat", data={"objective": "   "})
    assert res_empty.status_code == 400

    # 5. POST /api/chat with mock orchestrator
    mock_run_res = {
        "status": "success",
        "answer": "Simulated successful synthesis.",
        "telemetry": {"total_wall_time": 1.23, "agent_calls": 1},
        "trace": []
    }
    with patch("app.orchestrator.run", return_value=mock_run_res):
        res_chat = client.post("/api/chat", data={"objective": "Test prompt"})
        assert res_chat.status_code == 200
        assert res_chat.json()["status"] == "success"
        assert res_chat.json()["answer"] == "Simulated successful synthesis."

    print("       FastAPI endpoints (/api/status, /api/stop, /, /api/chat) validated.")

runner.run("7. FastAPI App & HTTP REST Endpoints", test_7_fastapi_endpoints)


# ─── Exit ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    code = runner.summary()
    sys.exit(code)
