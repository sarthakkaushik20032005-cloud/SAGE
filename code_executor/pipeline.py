"""
SAGE Code Executor — Pipeline Coordinator
==========================================
Orchestrates the complete code execution lifecycle:

    Step 1  Extract code from raw LLM response
    Step 2  First execution in Docker sandbox
    Step 3  Auto-fix loop (up to max_fix_attempts)
               ├─ Build repair prompt
               ├─ Call CoderInterface → get fixed code
               └─ Re-run in sandbox
    Step 4  Return PipelineResult (all details preserved)

Entry point: CodeExecutionPipeline.execute(task, llm_response)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .extractor import extract_code
from .fixer import CodeFixer, FixAttempt
from .sandbox import DockerSandbox, SandboxResult


# ─── Result ──────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Complete, auditable result of one code execution pipeline run.

    status values:
        "success"        — code ran and exited 0 (stderr alone is not failure)
        "error"          — code exited non-zero after all fix attempts
        "timeout"        — code was killed by the wall-clock limit
        "extract_failed" — no executable code found in the LLM response
        "infra_error"    — Docker infrastructure failure (daemon unavailable, etc.)
    """
    status: str

    # Final code (post-repair if any)
    final_code: str

    # Raw execution outputs
    stdout: str
    stderr: str
    exit_code: int
    wall_time_ms: float
    memory_peak_mb: float
    timed_out: bool

    # Attempt accounting
    attempts: int                               # 1 = first run only; 2+ = repairs used
    fix_history: List[FixAttempt] = field(default_factory=list)

    # Code extraction metadata
    extraction_had_fence: bool = False
    extraction_language: str = "unknown"

    # Infrastructure error detail (only set when status == "infra_error")
    infra_error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    def summary(self) -> str:
        """One-line human-readable summary for logs and telemetry."""
        if self.succeeded:
            return (
                f"SUCCESS | exit={self.exit_code} | "
                f"{self.wall_time_ms:.0f}ms | "
                f"{self.memory_peak_mb:.1f}MB | "
                f"attempts={self.attempts}"
            )
        return (
            f"FAILED [{self.status.upper()}] | exit={self.exit_code} | "
            f"timed_out={self.timed_out} | attempts={self.attempts}"
        )


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class CodeExecutionPipeline:
    """
    Coordinates code extraction → Docker sandbox execution → auto-fix → result.

    Args:
        sandbox:          DockerSandbox instance.
        fixer:            CodeFixer wrapping a CoderInterface.
        max_fix_attempts: Maximum LLM repair + re-run cycles after initial failure.
                          Set to 0 to disable auto-repair.
    """

    def __init__(
        self,
        sandbox: DockerSandbox,
        fixer: CodeFixer,
        max_fix_attempts: int = 2,
    ) -> None:
        self.sandbox = sandbox
        self.fixer = fixer
        self.max_fix_attempts = max_fix_attempts

    def execute(self, task: str, llm_response: str) -> PipelineResult:
        """
        Run the full pipeline.

        Args:
            task:         Original natural-language task (used in repair prompts).
            llm_response: Raw LLM output — may contain fences, prose, etc.

        Returns:
            PipelineResult with all execution and repair details preserved.
        """
        # ── Step 1: Extract code ─────────────────────────────────────────
        extracted = extract_code(llm_response)

        if not extracted.code.strip():
            return PipelineResult(
                status="extract_failed",
                final_code="",
                stdout="",
                stderr="No executable code block found in the LLM response.",
                exit_code=-1,
                wall_time_ms=0.0,
                memory_peak_mb=0.0,
                timed_out=False,
                attempts=0,
                extraction_had_fence=extracted.had_fence,
                extraction_language=extracted.language,
            )

        current_code = extracted.code
        fix_history: List[FixAttempt] = []

        # ── Step 2: First execution ──────────────────────────────────────
        sandbox_result = self.sandbox.run(current_code)

        if sandbox_result.error:
            # Docker infrastructure failure — do not attempt repair
            return PipelineResult(
                status="infra_error",
                final_code=current_code,
                stdout=sandbox_result.stdout,
                stderr=sandbox_result.stderr,
                exit_code=sandbox_result.exit_code,
                wall_time_ms=sandbox_result.wall_time_ms,
                memory_peak_mb=sandbox_result.memory_peak_mb,
                timed_out=sandbox_result.timed_out,
                attempts=1,
                infra_error=sandbox_result.error,
                extraction_had_fence=extracted.had_fence,
                extraction_language=extracted.language,
            )

        if sandbox_result.succeeded:
            return PipelineResult(
                status="success",
                final_code=current_code,
                stdout=sandbox_result.stdout,
                stderr=sandbox_result.stderr,
                exit_code=sandbox_result.exit_code,
                wall_time_ms=sandbox_result.wall_time_ms,
                memory_peak_mb=sandbox_result.memory_peak_mb,
                timed_out=False,
                attempts=1,
                extraction_had_fence=extracted.had_fence,
                extraction_language=extracted.language,
            )

        # ── Step 3: Auto-fix loop ────────────────────────────────────────
        for attempt_num in range(1, self.max_fix_attempts + 1):
            failing_code = current_code
            failing_result = sandbox_result

            # Build the repair prompt before fixing (needed for fix_history record)
            repair_prompt = self.fixer.build_repair_prompt(
                task, failing_code, failing_result, attempt_num
            )

            # Ask the coder for a fix
            fixed_code = self.fixer.fix(
                task=task,
                failing_code=failing_code,
                result=failing_result,
                attempt=attempt_num,
            )

            # Execute the fixed code
            new_result = self.sandbox.run(fixed_code)

            fix_history.append(
                FixAttempt(
                    attempt_number=attempt_num,
                    original_code=failing_code,
                    fixed_code=fixed_code,
                    sandbox_result=new_result,
                    repair_prompt=repair_prompt,
                )
            )

            current_code = fixed_code
            sandbox_result = new_result

            if new_result.succeeded:
                return PipelineResult(
                    status="success",
                    final_code=current_code,
                    stdout=new_result.stdout,
                    stderr=new_result.stderr,
                    exit_code=new_result.exit_code,
                    wall_time_ms=new_result.wall_time_ms,
                    memory_peak_mb=new_result.memory_peak_mb,
                    timed_out=False,
                    attempts=attempt_num + 1,
                    fix_history=fix_history,
                    extraction_had_fence=extracted.had_fence,
                    extraction_language=extracted.language,
                )

        # ── Step 4: All attempts exhausted ──────────────────────────────
        final_status = "timeout" if sandbox_result.timed_out else "error"
        return PipelineResult(
            status=final_status,
            final_code=current_code,
            stdout=sandbox_result.stdout,
            stderr=sandbox_result.stderr,
            exit_code=sandbox_result.exit_code,
            wall_time_ms=sandbox_result.wall_time_ms,
            memory_peak_mb=sandbox_result.memory_peak_mb,
            timed_out=sandbox_result.timed_out,
            attempts=self.max_fix_attempts + 1,
            fix_history=fix_history,
            extraction_had_fence=extracted.had_fence,
            extraction_language=extracted.language,
        )
