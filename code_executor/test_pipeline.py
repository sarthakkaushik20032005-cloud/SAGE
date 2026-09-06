#!/usr/bin/env python3
"""
SAGE Code Executor — Standalone Test Suite
==========================================
Validates the complete code execution pipeline without any LLM models.
All repair steps use MockCoder so Qwen2.5-Coder does NOT need to be loaded.

Usage (from the SAGE root directory):
    python code_executor/test_pipeline.py

Covered scenarios:
    T1  Code extractor - fences, python3, fallback
    T2  Successful execution + stdout capture
    T3  stderr capture (success even with stderr present)
    T4  Syntax error detection
    T5  Runtime error detection
    T6  Automatic code repair via MockCoder
    T7  Timeout enforcement (infinite loop killed)
    T8  Memory limit (OOM kill)
    T9  Network isolation (--network none)
    T10 Host filesystem isolation
    T11 extract_failed on empty LLM response
"""
from __future__ import annotations

import io
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Optional

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Make SAGE root importable when run directly ──────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from code_executor.extractor import extract_code
from code_executor.fixer import CodeFixer, MockCoder
from code_executor.pipeline import CodeExecutionPipeline, PipelineResult
from code_executor.sandbox import DockerSandbox, SandboxResult

# ─── Test configuration ───────────────────────────────────────────────────────

TEST_SANDBOX_CFG = {
    "image": "python:3.12-slim",
    "timeout_seconds": 6,   # Short for fast tests; override per-test as needed
    "memory_mb": 256,
    "cpu_cores": 1.0,
    "max_fix_attempts": 2,
}

# ─── Minimal test runner ──────────────────────────────────────────────────────

class _TestRunner:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self._failures: List[tuple[str, str]] = []

    def run(self, name: str, fn) -> None:
        width = 60
        pad = "-" * max(0, width - len(name))
        print(f"\n+-- {name} {pad}")
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            print(f"+-- PASSED  ({elapsed:.1f}s)")
            self.passed += 1
        except AssertionError as exc:
            elapsed = time.time() - t0
            msg = str(exc) or "AssertionError"
            print(f"+-- FAILED  ({elapsed:.1f}s): {msg}")
            self.failed += 1
            self._failures.append((name, msg))
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            msg = f"{type(exc).__name__}: {exc}"
            print(f"+-- ERROR   ({elapsed:.1f}s): {msg}")
            self.failed += 1
            self._failures.append((name, msg))

    def summary(self) -> None:
        total = self.passed + self.failed
        bar = "=" * 65
        print(f"\n{bar}")
        print("  SAGE Code Executor - Test Results")
        print(bar)
        print(f"  Total  : {total}")
        print(f"  Passed : {self.passed}  [PASS]")
        print(f"  Failed : {self.failed}  {'[FAIL]' if self.failed else '[OK]'}")
        if self._failures:
            print("\n  Failed tests:")
            for name, msg in self._failures:
                print(f"    * {name}")
                print(f"      {msg}")
        print(bar)

    def exit_code(self) -> int:
        return 0 if self.failed == 0 else 1


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sandbox(timeout_seconds: Optional[int] = None) -> DockerSandbox:
    cfg = dict(TEST_SANDBOX_CFG)
    if timeout_seconds is not None:
        cfg["timeout_seconds"] = timeout_seconds
    return DockerSandbox(cfg)


def _pipeline(
    max_fix_attempts: int = 2,
    mock_fixes: Optional[List[str]] = None,
    timeout_seconds: Optional[int] = None,
) -> CodeExecutionPipeline:
    sb = _sandbox(timeout_seconds)
    mock = MockCoder()
    if mock_fixes:
        for fix in mock_fixes:
            mock.add_fix_response(fix)
    fixer = CodeFixer(mock)
    return CodeExecutionPipeline(sandbox=sb, fixer=fixer, max_fix_attempts=max_fix_attempts)


def _fence(code: str, lang: str = "python") -> str:
    """Wrap code in a fenced block — simulates raw LLM response."""
    return f"```{lang}\n{code}\n```"


def _print_result(r: PipelineResult) -> None:
    print(f"   status        : {r.status}")
    print(f"   exit_code     : {r.exit_code}")
    print(f"   timed_out     : {r.timed_out}")
    print(f"   attempts      : {r.attempts}")
    print(f"   wall_time_ms  : {r.wall_time_ms:.0f} ms")
    print(f"   memory_mb     : {r.memory_peak_mb:.1f} MB")
    if r.stdout:
        print(f"   stdout        :\n{textwrap.indent(r.stdout[:400], '      ')}")
    if r.stderr:
        print(f"   stderr        :\n{textwrap.indent(r.stderr[:300], '      ')}")


def _print_sandbox(r: SandboxResult) -> None:
    print(f"   succeeded     : {r.succeeded}")
    print(f"   exit_code     : {r.exit_code}")
    print(f"   timed_out     : {r.timed_out}")
    print(f"   wall_time_ms  : {r.wall_time_ms:.0f} ms")
    print(f"   memory_mb     : {r.memory_peak_mb:.1f} MB")
    if r.stdout:
        print(f"   stdout        :\n{textwrap.indent(r.stdout[:400], '      ')}")
    if r.stderr:
        print(f"   stderr        :\n{textwrap.indent(r.stderr[:300], '      ')}")


# ─── Tests ────────────────────────────────────────────────────────────────────

runner = _TestRunner()

# ── T1: Code extractor ────────────────────────────────────────────────────────
def t1_extractor():
    cases = [
        # (raw_input, expect_fence, expect_lang, expect_snippet)
        ('```python\nprint("hello")\n```',          True,  "python",  'print("hello")'),
        ('```python3\nx = 1 + 1\nprint(x)\n```',    True,  "python",  "x = 1"),
        ('```py\nimport math\n```',                  True,  "python",  "import math"),
        ('Here:\n```\nfor i in range(3):\n    print(i)\n```\nDone.',
                                                     True,  "unknown", "for i"),
        ('x = 42\nprint(x)',                         False, "unknown", "x = 42"),
    ]
    for raw, exp_fence, exp_lang, exp_snip in cases:
        r = extract_code(raw)
        assert r.had_fence == exp_fence,  f"had_fence: {r.had_fence!r} != {exp_fence!r} | input={raw!r}"
        assert r.language  == exp_lang,   f"language:  {r.language!r} != {exp_lang!r}"
        assert exp_snip in r.code,        f"snippet '{exp_snip}' not in code: {r.code!r}"
    print("   Handles: ```python, ```python3, ```py, generic ```, no-fence fallback.")

runner.run("T1 — Code extractor (fences / fallback)", t1_extractor)


# ── T2: Successful execution + stdout capture ─────────────────────────────────
def t2_success_stdout():
    sb = _sandbox()
    code = textwrap.dedent("""\
        import math
        print("Hello, SAGE!")
        print(f"sqrt(144) = {math.sqrt(144):.1f}")
        print(f"pi = {math.pi:.5f}")
    """)
    r = sb.run(code)
    _print_sandbox(r)
    assert r.succeeded,               f"Expected success. exit={r.exit_code} stderr={r.stderr[:200]}"
    assert r.exit_code == 0
    assert not r.timed_out
    assert "Hello, SAGE!"      in r.stdout
    assert "sqrt(144) = 12.0"  in r.stdout
    assert "3.14159"           in r.stdout
    assert r.wall_time_ms > 0

runner.run("T2 — Successful execution + stdout capture", t2_success_stdout)


# ── T3: stderr capture — success despite stderr ───────────────────────────────
def t3_stderr_on_success():
    """
    CORRECTNESS: stderr alone must NOT mean failure.
    Python programs routinely write warnings/logging to stderr
    while exiting cleanly with code 0.
    """
    sb = _sandbox()
    code = textwrap.dedent("""\
        import sys
        print("stdout: all good")
        print("DEPRECATION WARNING: old API", file=sys.stderr)
        print("stdout: done")
    """)
    r = sb.run(code)
    _print_sandbox(r)
    assert r.succeeded,                  "stderr alone must not cause failure"
    assert r.exit_code == 0
    assert "stdout: all good"        in r.stdout
    assert "stdout: done"            in r.stdout
    assert "DEPRECATION WARNING"     in r.stderr
    print("   ✓ stderr captured separately; success NOT affected by stderr content.")

runner.run("T3 — stderr capture (success despite stderr)", t3_stderr_on_success)


# ── T4: Syntax error detection ────────────────────────────────────────────────
def t4_syntax_error():
    sb = _sandbox()
    code = textwrap.dedent("""\
        def broken(
            print("this will never parse"
    """)
    r = sb.run(code)
    _print_sandbox(r)
    assert not r.succeeded
    assert r.exit_code != 0
    assert not r.timed_out
    assert r.error is None                 # Infrastructure is fine; code is broken
    assert "SyntaxError" in r.stderr or "Error" in r.stderr
    print("   ✓ Syntax error → exit_code != 0, SyntaxError in stderr.")

runner.run("T4 — Syntax error detection", t4_syntax_error)


# ── T5: Runtime error detection ───────────────────────────────────────────────
def t5_runtime_error():
    sb = _sandbox()
    code = textwrap.dedent("""\
        data = {"key": "value"}
        print(data["missing_key"])   # KeyError
    """)
    r = sb.run(code)
    _print_sandbox(r)
    assert not r.succeeded
    assert r.exit_code != 0
    assert "KeyError" in r.stderr or "Error" in r.stderr
    print("   ✓ Runtime KeyError detected; exit_code != 0.")

runner.run("T5 — Runtime error detection", t5_runtime_error)


# ── T6: Automatic code repair via MockCoder ───────────────────────────────────
def t6_auto_repair():
    """
    Broken code → MockCoder provides a fix → successful execution.
    Verifies the complete fix loop without any live LLM.
    """
    broken_response = _fence(
        "print(undefined_variable_xyz)  # NameError — will fail on first run"
    )
    fixed_response = _fence(
        'print("auto-repaired by MockCoder!")\n'
        'print("pipeline fix loop works!")'
    )

    pipeline = _pipeline(max_fix_attempts=2, mock_fixes=[fixed_response])
    r = pipeline.execute(
        task="Print a success confirmation message",
        llm_response=broken_response,
    )
    _print_result(r)
    assert r.status   == "success", f"Expected success, got: {r.status}"
    assert r.attempts == 2,         f"Expected 2 attempts, got: {r.attempts}"
    assert len(r.fix_history) == 1, f"Expected 1 fix attempt, got: {len(r.fix_history)}"
    assert "auto-repaired"     in r.stdout
    assert "pipeline fix loop" in r.stdout
    print("   ✓ NameError detected → MockCoder fixed → ran successfully on attempt 2.")

runner.run("T6 — Automatic code repair (MockCoder fix loop)", t6_auto_repair)


# ── T7: Timeout enforcement ────────────────────────────────────────────────────
def t7_timeout():
    """
    An infinite loop must be killed within the configured wall-clock limit.
    max_fix_attempts=0 to avoid wasting time retrying a loop we know will timeout.
    """
    code = "while True: pass   # deliberate infinite loop"
    t0 = time.time()
    pipeline = _pipeline(max_fix_attempts=0, timeout_seconds=5)
    r = pipeline.execute(
        task="Run infinite loop",
        llm_response=_fence(code),
    )
    elapsed = time.time() - t0
    _print_result(r)
    print(f"   Actual wall time: {elapsed:.1f}s  (limit: 5s)")
    assert r.timed_out,                                 "Expected timed_out=True"
    assert r.status in ("timeout", "error"),            f"Unexpected status: {r.status}"
    assert elapsed < 5 + 8,                             f"Took too long: {elapsed:.1f}s"
    print(f"   ✓ Infinite loop killed after ~{elapsed:.1f}s.")

runner.run("T7 — Timeout enforcement (infinite loop killed)", t7_timeout)


# -- T8: Memory limit enforcement --
def t8_memory_limit():
    """
    Allocating more RAM than the container cap must fail.

    cgroup v1:       OOM kill is immediate -> exit 137, timed_out=False.
    WSL2/cgroup v2:  OOM kill can be slow; our wall-clock timeout may fire
                     first -> exit 137, timed_out=True.

    Both are acceptable. The decisive invariants are:
      1. not succeeded
      2. exit_code != 0
      3. The 'allocated' confirmation line is absent from stdout
    """
    limit_mb = TEST_SANDBOX_CFG["memory_mb"]
    exceed_mb = limit_mb + 150  # 150 MB over the cap

    code = textwrap.dedent(f"""\
        # Attempt to allocate {exceed_mb} MB -- exceeds {limit_mb} MB limit
        print("Attempting large allocation...")
        data = bytearray({exceed_mb} * 1024 * 1024)
        print("allocated -- this line must NOT appear if limit enforcement works")
    """)

    pipeline = _pipeline(max_fix_attempts=0)
    r = pipeline.execute(
        task="Allocate too much memory",
        llm_response=_fence(code),
    )
    _print_result(r)
    assert not r.succeeded, "Memory-exceeded code should NOT succeed"
    assert r.exit_code != 0, f"Exit code must be non-zero. Got: {r.exit_code}"
    # Both OOM kill (instant) and timeout (slow OOM on WSL2) prove limit was enforced
    assert r.timed_out or r.exit_code == 137, (
        f"Expected OOM kill (exit 137) or timeout. Got exit={r.exit_code} timed_out={r.timed_out}"
    )
    assert "allocated -- this line" not in r.stdout, (
        "Allocation success line appeared -- memory limit was NOT enforced!"
    )
    outcome = "timeout (OOM slow on WSL2)" if r.timed_out else f"OOM kill exit={r.exit_code}"
    print(f"   Memory limit enforced via: {outcome}. Allocation line absent from stdout.")

runner.run("T8 -- Memory limit enforcement (OOM kill)", t8_memory_limit)


# -- T9: Network isolation --
def t9_network_isolation():
    """
    The sandbox runs with --network none.

    Strategy: read /sys/class/net/ to check interfaces.
    With --network none, Docker creates ONLY the loopback (lo) interface.
    This check is instant and does not involve any blocking I/O.
    On WSL2, socket.create_connection() to an external IP can hang
    indefinitely when --network none prevents the kernel from returning
    ENETUNREACH promptly, so we avoid relying on socket timeouts.
    """
    sb = _sandbox()
    code = textwrap.dedent("""\
        import os
        import socket

        # 1. Interface check: --network none -> only 'lo' exists
        ifaces = os.listdir('/sys/class/net/')
        external = [i for i in ifaces if i != 'lo']
        print(f"Interfaces: {sorted(ifaces)}")
        if external:
            print(f"NETWORK_NOT_ISOLATED: external interfaces: {external}")
            raise SystemExit(1)
        print("INTERFACE_CHECK_OK: only loopback present")

        # 2. Loopback must be reachable (connects, gets refused = normal)
        try:
            socket.create_connection(('127.0.0.1', 9), timeout=1)
        except ConnectionRefusedError:
            print("LOOPBACK_OK: loopback reachable (port 9 refused, as expected)")
        except Exception as e:
            print(f"LOOPBACK_UNEXPECTED: {type(e).__name__}: {e}")

        # 3. External IP must be blocked (non-blocking check via raw socket)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(False)
        err = s.connect_ex(('8.8.8.8', 53))
        s.close()
        # EINPROGRESS (115) = async connect started but no route -> blocked
        # ENETUNREACH (101) = no route to host -> blocked immediately
        # ECONNREFUSED (111) would mean connected (bad), but can't happen with no route
        if err in (0,):  # 0 means connected immediately = NOT isolated
            print("EXTERNAL_REACHABLE -- isolation broken!")
            raise SystemExit(1)
        else:
            print(f"EXTERNAL_BLOCKED: connect_ex returned errno={err}")

        print("NETWORK_ISOLATED_OK")
    """)
    r = sb.run(code)
    _print_sandbox(r)
    assert r.succeeded, f"Test script crashed. exit={r.exit_code} stderr={r.stderr[:300]}"
    assert "INTERFACE_CHECK_OK"  in r.stdout, f"Interface check failed. stdout={r.stdout}"
    assert "NETWORK_ISOLATED_OK" in r.stdout, f"Isolation not confirmed. stdout={r.stdout}"
    assert "NETWORK_NOT_ISOLATED" not in r.stdout
    assert "EXTERNAL_REACHABLE"   not in r.stdout
    print("   Only loopback interface present; external IP non-routable; isolation confirmed.")

runner.run("T9 -- Network isolation (--network none)", t9_network_isolation)


# ── T10: Host filesystem isolation ────────────────────────────────────────────
def t10_filesystem_isolation():
    """
    The container must NOT expose any host filesystem paths.
    Verifies:
      • /sandbox/code.py is readable (bind-mount works)
      • /sandbox/code.py is NOT writable (read-only mount)
      • /sandbox/workspace is writable (tmpfs)
      • Windows/WSL2 host drive paths are not accessible
    """
    sb = _sandbox()
    code = textwrap.dedent("""\
        import os

        results = []

        # 1. Our code file should be readable (confirms bind-mount)
        try:
            lines = open("/sandbox/code.py").readlines()
            results.append(f"CODE_READABLE: {len(lines)} lines")
        except Exception as e:
            results.append(f"CODE_READ_FAILED: {e}")

        # 2. Code file must NOT be writable (read-only bind-mount)
        try:
            open("/sandbox/code.py", "w").write("TAMPERED")
            results.append("CODE_WRITABLE — ISOLATION BROKEN!")
        except OSError:
            results.append("CODE_READONLY_OK")

        # 3. Workspace tmpfs must be writable
        try:
            with open("/sandbox/workspace/test.txt", "w") as f:
                f.write("workspace write test")
            readback = open("/sandbox/workspace/test.txt").read()
            results.append(f"WORKSPACE_WRITABLE_OK: '{readback}'")
        except Exception as e:
            results.append(f"WORKSPACE_WRITE_FAILED: {e}")

        # 4. Host Windows/WSL2 paths must NOT be accessible
        host_paths_to_check = [
            "/mnt/c/Users",     # WSL2 Windows C: drive
            "/mnt/host",        # Possible WSL2 host mount
            "/host",
        ]
        host_visible = [p for p in host_paths_to_check if os.path.exists(p)
                        and os.listdir(p)]
        if host_visible:
            results.append(f"HOST_FS_VISIBLE: {host_visible}")
        else:
            results.append("HOST_FS_ISOLATED_OK")

        for r in results:
            print(r)

        broken = [r for r in results
                  if "BROKEN" in r or "FAILED" in r or "VISIBLE" in r]
        if broken:
            raise SystemExit(1)
    """)
    r = sb.run(code)
    _print_sandbox(r)
    assert r.succeeded,                            f"Test script crashed. stderr={r.stderr[:300]}"
    assert "CODE_READONLY_OK"      in r.stdout,    "Code file should be read-only"
    assert "WORKSPACE_WRITABLE_OK" in r.stdout,    "Workspace tmpfs should be writable"
    assert "HOST_FS_ISOLATED_OK"   in r.stdout,    f"Host FS visible! stdout={r.stdout}"
    assert "HOST_FS_VISIBLE"    not in r.stdout,   "Host FS must not be accessible"
    print("   ✓ Read-only code mount, writable workspace, host FS not accessible.")

runner.run("T10 — Host filesystem isolation", t10_filesystem_isolation)


# ── T11: extract_failed on empty response ─────────────────────────────────────
def t11_extract_failed():
    """An empty or whitespace-only LLM response must yield extract_failed."""
    for empty_input in ["   ", "", "\n\n", "No code here, just prose."]:
        p = _pipeline(max_fix_attempts=0)
        # For pure prose with no fences, the fallback treats it as code.
        # Only truly empty strings yield extract_failed.
        if not empty_input.strip():
            r = p.execute(task="anything", llm_response=empty_input)
            assert r.status == "extract_failed", (
                f"Expected extract_failed for {empty_input!r}, got: {r.status}"
            )
    print("   ✓ Empty/whitespace LLM responses yield status='extract_failed'.")

runner.run("T11 — extract_failed on empty LLM response", t11_extract_failed)


# ─── Final summary ────────────────────────────────────────────────────────────

runner.summary()
sys.exit(runner.exit_code())
