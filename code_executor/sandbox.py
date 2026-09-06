"""
SAGE Code Executor — Docker Sandbox
=====================================
Executes Python code inside an isolated Docker container.

Security properties enforced per container:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Network      │ none  (--network none)                          │
  │  Memory       │ configurable cap + swap disabled                │
  │  CPU          │ nano_cpus quota                                 │
  │  Capabilities │ ALL dropped  (cap_drop=ALL)                     │
  │  Privileges   │ no-new-privileges:true                          │
  │  PIDs         │ pids_limit=64  (fork-bomb protection)           │
  │  Signals      │ docker-init as PID 1  (init=True)               │
  │  Code mount   │ bind-mounted read-only at /sandbox/code.py      │
  │  Workspace    │ writable tmpfs at /sandbox/workspace (64 MB)    │
  │  Host FS      │ zero host paths mounted; no SAGE dir exposure   │
  └─────────────────────────────────────────────────────────────────┘

Docker is the ONLY supported backend.
If Docker is unavailable, a clear infrastructure error is returned.
No subprocess / local-execution fallback exists.
"""
from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import docker
import docker.errors


# ─── Result type ─────────────────────────────────────────────────────────────

@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    wall_time_ms: float
    memory_peak_mb: float
    timed_out: bool
    error: Optional[str] = None  # Set on Docker infrastructure errors only

    @property
    def succeeded(self) -> bool:
        """
        True iff the code exited cleanly.

        Rules:
          • exit_code == 0       — Python convention for success
          • not timed_out        — was not killed by the wall-clock limit
          • error is None        — no Docker infrastructure failure

        IMPORTANT: stderr alone does NOT constitute failure.
        Warnings, logging, and deprecation notices all write to stderr
        while the process exits 0.
        """
        return self.exit_code == 0 and not self.timed_out and self.error is None


# ─── Sandbox ─────────────────────────────────────────────────────────────────

class DockerSandbox:
    """Docker-only Python code execution sandbox."""

    def __init__(self, sandbox_config: Dict[str, Any]) -> None:
        """
        Args:
            sandbox_config: dict matching config.SANDBOX keys:
                image, timeout_seconds, memory_mb, cpu_cores
        """
        self.cfg = sandbox_config

    # ── Public API ────────────────────────────────────────────────────────

    def pull_image_if_missing(self) -> None:
        """Pull the sandbox base image if not cached locally."""
        client = self._client()
        image = self.cfg["image"]
        try:
            client.images.get(image)
            print(f"[Sandbox] Image '{image}' already present locally.")
        except docker.errors.ImageNotFound:
            print(f"[Sandbox] Pulling '{image}' — first-time setup, please wait…")
            for chunk in client.api.pull(image, stream=True, decode=True):
                status = chunk.get("status", "")
                progress = chunk.get("progress", "")
                if status:
                    print(f"  {status} {progress}", end="\r", flush=True)
            print(f"\n[Sandbox] '{image}' ready.            ")

    def run(self, code: str) -> SandboxResult:
        """
        Execute *code* in an isolated Docker container.

        Returns SandboxResult with stdout, stderr, timing, and memory.
        Never falls back to local execution.

        Raises RuntimeError only if Docker itself is completely unreachable
        (caught and converted to SandboxResult.error in practice).
        """
        # ── Verify Docker reachability ────────────────────────────────────
        try:
            client = self._client()
        except RuntimeError as exc:
            return SandboxResult(
                stdout="", stderr="", exit_code=-1,
                wall_time_ms=0.0, memory_peak_mb=0.0,
                timed_out=False, error=str(exc),
            )

        timeout_s = float(self.cfg["timeout_seconds"])
        memory_mb = int(self.cfg["memory_mb"])
        cpu_cores = float(self.cfg["cpu_cores"])
        image = str(self.cfg["image"])

        # ── Write code to an isolated temp file ───────────────────────────
        tmp_dir = Path(tempfile.mkdtemp(prefix="sage_sbx_"))
        code_file = tmp_dir / "code.py"
        try:
            code_file.write_text(code, encoding="utf-8")
        except OSError as exc:
            _cleanup_tmp(tmp_dir, code_file)
            return SandboxResult(
                stdout="", stderr="", exit_code=-1,
                wall_time_ms=0.0, memory_peak_mb=0.0,
                timed_out=False,
                error=f"Could not write code to temp file: {exc}",
            )

        container = None
        timed_out = False
        start_time = time.time()

        try:
            container = client.containers.run(
                image=image,
                # -u = unbuffered stdout/stderr for reliable log collection
                command=["python", "-u", "/sandbox/code.py"],
                volumes={
                    # Code mounted read-only — container cannot modify it
                    str(code_file.resolve()): {
                        "bind": "/sandbox/code.py",
                        "mode": "ro",
                    }
                },
                # Writable in-memory filesystems — zero host path exposure
                tmpfs={
                    "/sandbox/workspace": "size=64m",
                    "/tmp": "size=32m",
                },
                # ── Isolation settings ──────────────────────────────────
                network_mode="none",                # Full network isolation
                mem_limit=f"{memory_mb}m",
                memswap_limit=f"{memory_mb}m",      # Disable swap entirely
                nano_cpus=int(cpu_cores * 1_000_000_000),
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                pids_limit=64,                       # Fork-bomb protection
                init=True,                           # Proper signal handling
                working_dir="/sandbox/workspace",
                detach=True,
                auto_remove=False,
                stdout=True,
                stderr=True,
            )

            # ── Wait with Python-level thread timeout ─────────────────────
            # Using a thread is more reliable than docker-py's socket timeout
            # across different Docker/requests versions.
            exit_holder: list = [-1]
            wait_exc_holder: list = [None]

            def _wait_for_container() -> None:
                try:
                    result = container.wait()
                    exit_holder[0] = result.get("StatusCode", -1)
                except Exception as exc:  # noqa: BLE001
                    wait_exc_holder[0] = exc

            t = threading.Thread(target=_wait_for_container, daemon=True)
            t.start()
            t.join(timeout=timeout_s)

            if t.is_alive():
                # Wall-clock limit exceeded — kill the container
                timed_out = True
                try:
                    container.kill()
                except Exception:  # noqa: BLE001
                    pass
                t.join(timeout=5)   # Wait for wait-thread to unblock after kill

            exit_code: int = exit_holder[0]
            wall_time_ms = (time.time() - start_time) * 1000.0

            # ── Memory stats (best-effort) ────────────────────────────────
            memory_peak_mb = _extract_memory_mb(container)

            # ── Collect stdout / stderr ───────────────────────────────────
            stdout_str, stderr_str = _collect_logs(container)

            return SandboxResult(
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=exit_code,
                wall_time_ms=round(wall_time_ms, 2),
                memory_peak_mb=memory_peak_mb,
                timed_out=timed_out,
                error=None,
            )

        except (docker.errors.DockerException, Exception) as exc:
            wall_time_ms = (time.time() - start_time) * 1000.0
            return SandboxResult(
                stdout="", stderr="", exit_code=-1,
                wall_time_ms=round(wall_time_ms, 2), memory_peak_mb=0.0,
                timed_out=False,
                error=f"Docker infrastructure error: {exc}",
            )

        finally:
            # Always clean up — container first, then temp files.
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:  # noqa: BLE001
                    pass
            _cleanup_tmp(tmp_dir, code_file)

    # ── Private helpers ───────────────────────────────────────────────────

    def _client(self) -> docker.DockerClient:
        """Return a live Docker client or raise RuntimeError with a clear message."""
        try:
            client = docker.from_env(timeout=120)
            client.ping()
            return client
        except (docker.errors.DockerException, Exception) as exc:
            raise RuntimeError(
                "Docker is unavailable. "
                "Ensure Docker Desktop is running and accessible. "
                f"Details: {exc}"
            ) from exc


# ─── Module-level helpers ─────────────────────────────────────────────────────

def _extract_memory_mb(container) -> float:
    """
    Best-effort peak memory in MB.

    cgroup v1: memory_stats.max_usage  (explicit high-water mark)
    cgroup v2: memory_stats.usage      (snapshot at exit — approximation)
    """
    try:
        stats = container.stats(stream=False)
        mem = stats.get("memory_stats", {})
        peak_bytes = mem.get("max_usage") or mem.get("usage", 0)
        return round(peak_bytes / (1024 * 1024), 2)
    except Exception:  # noqa: BLE001
        return 0.0


def _collect_logs(container) -> Tuple[str, str]:
    """Retrieve separated stdout and stderr byte streams from the container."""
    try:
        out = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace").strip()
        err = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace").strip()
        return out, err
    except Exception:  # noqa: BLE001
        return "", ""


def _cleanup_tmp(tmp_dir: Path, code_file: Path) -> None:
    try:
        code_file.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        tmp_dir.rmdir()
    except Exception:  # noqa: BLE001
        pass
