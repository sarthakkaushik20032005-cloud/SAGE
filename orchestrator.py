import os
import re
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from model_manager import model_manager
from model_client import model_client
from document_processor import DocumentProcessor
from code_executor.pipeline import CodeExecutionPipeline
from code_executor.fixer import CodeFixer, RealCoder
from code_executor.sandbox import DockerSandbox

# Initialize UTF-8 safe Rich Console
console = Console(highlight=False, legacy_windows=False)

class Orchestrator:
    def __init__(self):
        self.agent_system_prompt = self._load_prompt("agent_system.txt")
        self.coder_system_prompt = self._load_prompt("coder_system.txt")
        self.document_system_prompt = self._load_prompt("document_system.txt")

        # Code Execution Pipeline (Docker sandbox + LLM auto-repair)
        _sandbox = DockerSandbox(config.SANDBOX)
        _real_coder = RealCoder(model_client, model_manager)
        _fixer = CodeFixer(_real_coder)
        self._code_pipeline = CodeExecutionPipeline(
            sandbox=_sandbox,
            fixer=_fixer,
            max_fix_attempts=config.SANDBOX["max_fix_attempts"]
        )

    def _load_prompt(self, filename: str) -> str:
        path = config.PROMPTS_DIR / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def _clean_json_str(self, text: str) -> str:
        text = text.strip()
        # 1. Strip reasoning / thought tags
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()

        # 2. Extract from markdown code fence anywhere in text
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate

        # 3. Extract JSON object containing "type": "tool_calls" | "final"
        obj_match = re.search(r"(\{\s*\"type\"\s*:\s*\"(?:tool_calls|final)\".*?\})", text, re.DOTALL)
        if obj_match:
            return obj_match.group(1).strip()

        # 4. Fallback to balanced outermost braces
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx + 1].strip()

        return text

    def _parse_gemma_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        # Try direct parse first
        try:
            data = json.loads(raw_text.strip())
            if isinstance(data, dict) and "type" in data:
                if data["type"] in ["tool_calls", "final"]:
                    return data
        except Exception:
            pass

        # Clean and try again
        cleaned = self._clean_json_str(raw_text)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "type" in data:
                if data["type"] in ["tool_calls", "final"]:
                    return data
        except Exception:
            pass
        return None

    def _call_coder(self, task: str, context: str, telemetry: Dict[str, Any], trace: List[Dict[str, Any]]) -> str:
        model_manager.ensure_model("coder")
        telemetry["coder_calls"] += 1

        coder_cfg = config.MODELS["coder"]
        messages = [
            {"role": "system", "content": self.coder_system_prompt},
            {"role": "user", "content": f"TASK:\n{task}\n\nCONTEXT / SPECIFICATIONS:\n{context}"}
        ]

        console.print(Panel(
            f"[bold green]TASK:[/bold green] {task}\n[dim]CONTEXT:[/dim] {context[:300]}...",
            title="[bold yellow]INPUT -> QWEN2.5-CODER[/bold yellow]",
            border_style="yellow"
        ))

        res = model_client.chat_completion(
            messages=messages,
            temperature=coder_cfg.get("temperature", 0.10),
            max_tokens=coder_cfg.get("max_tokens", 4096)
        )

        content = res["content"]
        duration = res["duration"]
        usage = res.get("usage", {})
        tok_info = f"{duration:.2f}s | {usage.get('completion_tokens', 'N/A')} tokens"

        console.print(Panel(
            content[:600] + ("..." if len(content) > 600 else ""),
            title=f"[bold yellow]OUTPUT <- QWEN2.5-CODER ({tok_info})[/bold yellow]",
            border_style="yellow"
        ))

        trace.append({
            "actor": "coder",
            "action": "code_generated",
            "task": task,
            "duration": duration,
            "snippet": content[:200] + "..." if len(content) > 200 else content
        })

        return content

    def _call_document_model(self, task_or_prompt: str, image_meta: Optional[Dict[str, Any]], telemetry: Dict[str, Any], trace: List[Dict[str, Any]]) -> str:
        model_manager.ensure_model("document_analyzer")
        telemetry["document_calls"] += 1

        doc_cfg = config.MODELS["document_analyzer"]
        
        if image_meta:
            input_desc = image_meta.get("description", "<image>")
            user_content = [
                {"type": "text", "text": f"{self.document_system_prompt}\n\nTask: {task_or_prompt}"},
                {"type": "image_url", "image_url": {"url": image_meta["data_uri"]}}
            ]
            console.print(Panel(
                f"[bold cyan]Task:[/bold cyan] {task_or_prompt}\n[bold cyan]Input:[/bold cyan] {input_desc}",
                title="[bold magenta]INPUT -> QWEN3-VL (Document / OCR)[/bold magenta]",
                border_style="magenta"
            ))
        else:
            input_desc = f"<text content: {len(task_or_prompt)} chars>"
            user_content = f"{self.document_system_prompt}\n\n{task_or_prompt}"
            console.print(Panel(
                f"[bold cyan]Task/Prompt:[/bold cyan] {task_or_prompt[:300]}...",
                title="[bold magenta]INPUT -> QWEN3-VL (Document Analyzer)[/bold magenta]",
                border_style="magenta"
            ))

        messages = [
            {"role": "user", "content": user_content}
        ]

        res = model_client.chat_completion(
            messages=messages,
            temperature=doc_cfg.get("temperature", 0.05),
            max_tokens=doc_cfg.get("max_tokens", 2048)
        )

        content = res["content"]
        duration = res["duration"]
        usage = res.get("usage", {})
        tok_info = f"{duration:.2f}s | {usage.get('completion_tokens', 'N/A')} tokens"

        console.print(Panel(
            content[:600] + ("..." if len(content) > 600 else ""),
            title=f"[bold magenta]OUTPUT <- QWEN3-VL ({tok_info})[/bold magenta]",
            border_style="magenta"
        ))

        trace.append({
            "actor": "document_analyzer",
            "action": "analyzed_content",
            "input": input_desc,
            "duration": duration,
            "snippet": content[:200] + "..." if len(content) > 200 else content
        })

        return content

    def run(
        self,
        user_objective: str,
        attachments_manifest: List[Dict[str, Any]],
        file_map: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        wall_start = time.time()
        telemetry = {
            "agent_calls": 0,
            "coder_calls": 0,
            "document_calls": 0,
            "model_switches_at_start": model_manager.switch_count,
            "total_wall_time": 0.0,
            "final_response_length": 0
        }
        trace: List[Dict[str, Any]] = []

        console.print("\n" + "="*70, style="bold cyan")
        console.print("[bold cyan][START] SAGE SEQUENTIAL MULTI-MODEL ORCHESTRATION[/bold cyan]")
        console.print("="*70 + "\n", style="bold cyan")
        console.print(f"[bold]User Objective:[/bold] {user_objective}")
        if attachments_manifest:
            console.print(f"[bold]Attachments:[/bold] {len(attachments_manifest)} registered file(s)")
            for att in attachments_manifest:
                console.print(f"  * [yellow]{att['ref']}[/yellow]: {att['name']} ({att['type']}, {att['size']} bytes)")

        # Initial conversation for Gemma
        user_payload = f"Objective:\n{user_objective}\n\nATTACHMENTS:\n{json.dumps(attachments_manifest, indent=2)}"
        history: List[Dict[str, Any]] = [
            {"role": "system", "content": self.agent_system_prompt},
            {"role": "user", "content": user_payload}
        ]

        final_answer = None
        loop_count = 0

        while loop_count < config.MAX_AGENT_LOOPS:
            loop_count += 1
            console.print("\n" + "="*60, style="bold blue")
            console.print(f"[bold blue]AGENT LOOP {loop_count} / {config.MAX_AGENT_LOOPS}[/bold blue]", style="bold blue")
            console.print("="*60, style="bold blue")

            # Ensure Gemma is loaded
            model_manager.ensure_model("agent")
            telemetry["agent_calls"] += 1
            agent_cfg = config.MODELS["agent"]

            # Terminal log input to Gemma
            last_msg = history[-1]
            console.print(Panel(
                f"[dim]Role: {last_msg['role']}[/dim]\n{last_msg['content'][:500]}...",
                title=f"[bold blue]INPUT -> GEMMA (Loop {loop_count})[/bold blue]",
                border_style="blue"
            ))

            # Call Gemma
            res = model_client.chat_completion(
                messages=history,
                temperature=agent_cfg.get("temperature", 0.20),
                max_tokens=agent_cfg.get("max_tokens", 2048)
            )

            raw_output = res["content"]
            duration = res["duration"]
            usage = res.get("usage", {})
            tok_info = f"{duration:.2f}s | {usage.get('completion_tokens', 'N/A')} tokens"

            console.print(Panel(
                raw_output,
                title=f"[bold blue]OUTPUT <- GEMMA ({tok_info})[/bold blue]",
                border_style="blue"
            ))

            # Parse JSON
            parsed = self._parse_gemma_json(raw_output)

            # Defensive 1-turn repair if JSON is invalid
            if parsed is None:
                console.print("[bold red][WARN] Gemma output was not valid JSON. Attempting 1 repair turn...[/bold red]")
                repair_prompt = (
                    "Your previous response was not valid JSON matching the required schema.\n"
                    "Output ONLY a single valid JSON object:\n"
                    "Either:\n"
                    "{\"type\": \"tool_calls\", \"calls\": [...]}\n"
                    "or:\n"
                    "{\"type\": \"final\", \"answer\": \"...\"}\n"
                    "No prose outside JSON."
                )
                history.append({"role": "assistant", "content": raw_output})
                history.append({"role": "user", "content": repair_prompt})

                res = model_client.chat_completion(
                    messages=history,
                    temperature=0.10,
                    max_tokens=agent_cfg.get("max_tokens", 2048)
                )
                raw_output = res["content"]
                parsed = self._parse_gemma_json(raw_output)
                
                # Remove repair turn from history before proceeding
                history.pop()
                history.pop()

                if parsed is None:
                    raise RuntimeError(f"Gemma failed to produce valid JSON after repair attempt. Raw: {raw_output}")

            # Process Response
            res_type = parsed.get("type")

            if res_type == "final":
                final_answer = parsed.get("answer", "")
                trace.append({
                    "actor": "gemma",
                    "action": "final_synthesis",
                    "loop": loop_count,
                    "answer_preview": final_answer[:200] + "..." if len(final_answer) > 200 else final_answer
                })
                break

            elif res_type == "tool_calls":
                calls = parsed.get("calls", [])
                if not calls:
                    raise RuntimeError("Gemma returned tool_calls with an empty 'calls' array.")

                trace.append({
                    "actor": "gemma",
                    "action": "requested_tools",
                    "loop": loop_count,
                    "calls": calls
                })

                # Append Gemma's assistant turn
                history.append({"role": "assistant", "content": json.dumps(parsed, indent=2)})

                tool_results_list = []

                for call in calls:
                    tool_name = call.get("tool")
                    task = call.get("task", "")

                    if tool_name == "document_analyzer":
                        input_refs = call.get("input_refs", [])
                        if not input_refs:
                            input_refs = [attachments_manifest[0]["ref"]] if attachments_manifest else []

                        doc_results = []
                        for ref in input_refs:
                            if ref not in file_map:
                                doc_results.append(f"Error: Referenced file '{ref}' is not registered.")
                                continue

                            finfo = file_map[ref]
                            fpath = finfo["path"]
                            ftype = finfo["type"]

                            console.print(Panel(
                                f"[bold]Tool:[/bold] document_analyzer\n[bold]File:[/bold] {ref} ({finfo['name']})\n[bold]Task:[/bold] {task}",
                                title="[bold cyan]EXECUTING TOOL: DOCUMENT ANALYZER[/bold cyan]",
                                border_style="cyan"
                            ))

                            def doc_callback(prompt_or_task: str, meta: Optional[Dict[str, Any]]) -> str:
                                return self._call_document_model(prompt_or_task, meta, telemetry, trace)

                            doc_res = DocumentProcessor.process(
                                filepath=fpath,
                                task=task,
                                file_type=ftype,
                                call_model_fn=doc_callback
                            )
                            doc_results.append(f"[{ref} Analysis]:\n{doc_res}")

                        tool_results_list.append({
                            "tool": "document_analyzer",
                            "status": "success",
                            "input_refs": input_refs,
                            "result": "\n\n".join(doc_results)
                        })

                    elif tool_name == "coder":
                        context = call.get("context", "")
                        console.print(Panel(
                            f"[bold]Tool:[/bold] coder + sandbox\n[bold]Task:[/bold] {task}\n[bold]Context:[/bold] {context[:200]}...",
                            title="[bold yellow]EXECUTING TOOL: CODER + SANDBOX[/bold yellow]",
                            border_style="yellow"
                        ))

                        # 1. Call Qwen2.5-Coder → raw LLM response
                        raw_llm_response = self._call_coder(task, context, telemetry, trace)

                        # 2. Extract → sandbox → auto-fix loop
                        pipe_result = self._code_pipeline.execute(
                            task=task,
                            llm_response=raw_llm_response
                        )
                        telemetry["sandbox_executions"] = (
                            telemetry.get("sandbox_executions", 0) + 1
                        )

                        # 3. Log the sandbox result
                        exec_summary = (
                            f"[{'SUCCESS' if pipe_result.succeeded else pipe_result.status.upper()}] "
                            f"exit={pipe_result.exit_code} | "
                            f"time={pipe_result.wall_time_ms:.0f}ms | "
                            f"mem={pipe_result.memory_peak_mb:.1f}MB | "
                            f"attempts={pipe_result.attempts}"
                        )
                        if pipe_result.stdout:
                            exec_summary += f"\n\nSTDOUT:\n{pipe_result.stdout[:500]}"
                        if pipe_result.stderr and not pipe_result.succeeded:
                            exec_summary += f"\n\nSTDERR:\n{pipe_result.stderr[:300]}"

                        console.print(Panel(
                            exec_summary,
                            title="[bold green]SANDBOX EXECUTION RESULT[/bold green]"
                            if pipe_result.succeeded
                            else "[bold red]SANDBOX EXECUTION FAILED[/bold red]",
                            border_style="green" if pipe_result.succeeded else "red"
                        ))

                        trace.append({
                            "actor": "sandbox",
                            "action": "executed_code",
                            "status": pipe_result.status,
                            "exit_code": pipe_result.exit_code,
                            "wall_time_ms": pipe_result.wall_time_ms,
                            "memory_peak_mb": pipe_result.memory_peak_mb,
                            "attempts": pipe_result.attempts,
                            "stdout_preview": pipe_result.stdout[:200] if pipe_result.stdout else "",
                        })

                        # 4. Return rich result to Gemma (includes stdout, code, stats)
                        tool_results_list.append({
                            "tool": "coder",
                            "status": pipe_result.status,
                            "code": pipe_result.final_code,
                            "stdout": pipe_result.stdout,
                            "stderr": pipe_result.stderr if not pipe_result.succeeded else "",
                            "execution_time_ms": pipe_result.wall_time_ms,
                            "memory_mb": pipe_result.memory_peak_mb,
                            "attempts": pipe_result.attempts,
                        })

                    else:
                        tool_results_list.append({
                            "tool": tool_name or "unknown",
                            "status": "error",
                            "result": f"Unknown tool '{tool_name}'. Available tools: document_analyzer, coder"
                        })

                tool_result_payload = {
                    "type": "tool_results",
                    "results": tool_results_list
                }

                console.print(Panel(
                    json.dumps(tool_result_payload, indent=2)[:800] + ("..." if len(json.dumps(tool_result_payload)) > 800 else ""),
                    title="[bold green]RETURNING TOOL RESULTS TO GEMMA[/bold green]",
                    border_style="green"
                ))

                # Append runtime tool results to agent history
                history.append({"role": "user", "content": json.dumps(tool_result_payload, indent=2)})

            else:
                raise RuntimeError(f"Unexpected response type from Gemma: {res_type}")

        if final_answer is None:
            raise RuntimeError(f"Maximum agent loops ({config.MAX_AGENT_LOOPS}) reached without generating a final response.")

        wall_end = time.time()
        telemetry["total_wall_time"] = wall_end - wall_start
        telemetry["final_response_length"] = len(final_answer)
        telemetry["model_switches"] = model_manager.switch_count - telemetry["model_switches_at_start"]

        # Final Telemetry Summary Table
        table = Table(title="[bold green]SAGE EXECUTION TELEMETRY[/bold green]", border_style="green")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="bold white")

        table.add_row("Agent (Gemma) Calls", str(telemetry["agent_calls"]))
        table.add_row("Document Analyzer (Qwen3-VL) Calls", str(telemetry["document_calls"]))
        table.add_row("Coder (Qwen2.5-Coder) Calls", str(telemetry["coder_calls"]))
        table.add_row("Model Switches", str(telemetry["model_switches"]))
        table.add_row("Total Wall Time", f"{telemetry['total_wall_time']:.2f} seconds")
        table.add_row("Final Answer Length", f"{telemetry['final_response_length']} chars")

        console.print("\n" + "="*60, style="bold green")
        console.print("[bold green][SUCCESS] REQUEST COMPLETE[/bold green]")
        console.print("="*60, style="bold green")
        console.print(table)
        console.print("="*60 + "\n", style="bold green")

        return {
            "status": "success",
            "answer": final_answer,
            "telemetry": telemetry,
            "trace": trace
        }

orchestrator = Orchestrator()
