// SAGE Web Client Logic
document.addEventListener("DOMContentLoaded", () => {
    const chatHistory = document.getElementById("chatHistory");
    const promptInput = document.getElementById("promptInput");
    const sendBtn = document.getElementById("sendBtn");
    const attachBtn = document.getElementById("attachBtn");
    const fileInput = document.getElementById("fileInput");
    const attachmentsTray = document.getElementById("attachmentsTray");
    const dropZoneOverlay = document.getElementById("dropZoneOverlay");
    const currentModelLabel = document.getElementById("currentModelLabel");

    // Telemetry Elements
    const telemetryCard = document.getElementById("telemetryCard");
    const metricAgentCalls = document.getElementById("metricAgentCalls");
    const metricDocCalls = document.getElementById("metricDocCalls");
    const metricCoderCalls = document.getElementById("metricCoderCalls");
    const metricSwitches = document.getElementById("metricSwitches");
    const metricWallTime = document.getElementById("metricWallTime");
    const traceTimeline = document.getElementById("traceTimeline");

    let attachedFiles = [];

    // Auto status poll
    async function updateStatus() {
        try {
            const res = await fetch("/api/status");
            if (res.ok) {
                const data = await res.json();
                if (data.current_model) {
                    const name = data.models[data.current_model] || data.current_model;
                    currentModelLabel.textContent = `Active: ${name}`;
                } else {
                    currentModelLabel.textContent = "System Ready (Sequential)";
                }
            }
        } catch (e) {
            currentModelLabel.textContent = "Offline";
        }
    }
    setInterval(updateStatus, 4000);
    updateStatus();

    // Attach File Trigger
    attachBtn.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        handleFiles(Array.from(e.target.files));
        fileInput.value = "";
    });

    // Clipboard Paste for Images
    window.addEventListener("paste", (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        const pastedFiles = [];
        for (let item of items) {
            if (item.kind === "file") {
                const file = item.getAsFile();
                if (file) {
                    const ext = file.type.split("/")[1] || "png";
                    const renamed = new File([file], `clipboard_${Date.now()}.${ext}`, { type: file.type });
                    pastedFiles.push(renamed);
                }
            }
        }
        if (pastedFiles.length > 0) {
            handleFiles(pastedFiles);
        }
    });

    // Drag and Drop
    window.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZoneOverlay.classList.add("active");
    });

    window.addEventListener("dragleave", (e) => {
        if (e.clientX <= 0 || e.clientY <= 0) {
            dropZoneOverlay.classList.remove("active");
        }
    });

    window.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZoneOverlay.classList.remove("active");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFiles(Array.from(e.dataTransfer.files));
        }
    });

    function handleFiles(files) {
        files.forEach((file) => {
            // Avoid exact duplicate additions
            if (!attachedFiles.some(f => f.name === file.name && f.size === file.size)) {
                attachedFiles.push(file);
            }
        });
        renderAttachments();
    }

    function renderAttachments() {
        if (attachedFiles.length === 0) {
            attachmentsTray.style.display = "none";
            attachmentsTray.innerHTML = "";
            return;
        }

        attachmentsTray.style.display = "flex";
        attachmentsTray.innerHTML = "";

        attachedFiles.forEach((file, index) => {
            const pill = document.createElement("div");
            pill.className = "attachment-pill";

            const isImage = file.type.startsWith("image/");
            if (isImage) {
                const img = document.createElement("img");
                img.className = "attachment-thumb";
                img.src = URL.createObjectURL(file);
                pill.appendChild(img);
            }

            const info = document.createElement("span");
            const sizeKb = (file.size / 1024).toFixed(1);
            info.textContent = `${file.name} (${sizeKb} KB)`;
            pill.appendChild(info);

            const removeBtn = document.createElement("button");
            removeBtn.className = "remove-btn";
            removeBtn.innerHTML = "×";
            removeBtn.title = "Remove";
            removeBtn.addEventListener("click", () => {
                attachedFiles.splice(index, 1);
                renderAttachments();
            });
            pill.appendChild(removeBtn);

            attachmentsTray.appendChild(pill);
        });
    }

    // Auto-resize textarea
    promptInput.addEventListener("input", () => {
        promptInput.style.height = "auto";
        promptInput.style.height = Math.min(promptInput.scrollHeight, 150) + "px";
    });

    // Enter key to send (Shift+Enter for newline)
    promptInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener("click", sendMessage);

    async function sendMessage() {
        const text = promptInput.value.trim();
        if (!text && attachedFiles.length === 0) return;

        // Freeze input
        sendBtn.disabled = true;
        promptInput.disabled = true;
        sendBtn.innerHTML = `<span>Running...</span>`;

        // Render User Bubble
        appendUserMessage(text, attachedFiles);

        // Prepare FormData
        const formData = new FormData();
        formData.append("objective", text || "Analyze the attached files and perform the required operations.");
        attachedFiles.forEach((file) => {
            formData.append("files", file);
        });

        // Clear input and attachments
        const sendingFiles = [...attachedFiles];
        attachedFiles = [];
        renderAttachments();
        promptInput.value = "";
        promptInput.style.height = "auto";

        // Show thinking indicator
        const thinkingBubble = appendThinkingMessage();

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            thinkingBubble.remove();

            if (!response.ok || data.status === "error") {
                appendErrorMessage(data.error || "An error occurred during multi-model orchestration.");
                if (data.traceback) {
                    console.error("Backend Traceback:", data.traceback);
                }
            } else {
                appendSageResponse(data.answer, data.telemetry);
                updateTelemetryAndTrace(data.telemetry, data.trace);
            }
        } catch (err) {
            thinkingBubble.remove();
            appendErrorMessage(`Network / Connection Error: ${err.message}`);
        } finally {
            sendBtn.disabled = false;
            promptInput.disabled = false;
            sendBtn.innerHTML = `<span>Send</span><span class="send-icon">➤</span>`;
            promptInput.focus();
            updateStatus();
        }
    }

    function appendUserMessage(text, files) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "chat-msg user";

        const header = document.createElement("div");
        header.className = "msg-header";
        header.textContent = "You";
        msgDiv.appendChild(header);

        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        bubble.textContent = text;

        if (files && files.length > 0) {
            const attachContainer = document.createElement("div");
            attachContainer.className = "msg-attachments";
            files.forEach(f => {
                const chip = document.createElement("div");
                chip.className = "msg-file-chip";
                chip.textContent = `📎 ${f.name} (${(f.size/1024).toFixed(1)} KB)`;
                attachContainer.appendChild(chip);
            });
            bubble.appendChild(attachContainer);
        }

        msgDiv.appendChild(bubble);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function appendThinkingMessage() {
        const msgDiv = document.createElement("div");
        msgDiv.className = "chat-msg sage";

        const header = document.createElement("div");
        header.className = "msg-header";
        header.textContent = "SAGE Orchestrator";
        msgDiv.appendChild(header);

        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        bubble.innerHTML = `<em>🧠 Gemma is reasoning, coordinating specialists, and switching models...</em>`;

        msgDiv.appendChild(bubble);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return msgDiv;
    }

    function appendSageResponse(answer, telemetry) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "chat-msg sage";

        const header = document.createElement("div");
        header.className = "msg-header";
        const timeStr = telemetry ? `(${telemetry.total_wall_time.toFixed(1)}s)` : "";
        header.textContent = `SAGE ${timeStr}`;
        msgDiv.appendChild(header);

        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        bubble.innerHTML = formatMarkdown(answer);

        msgDiv.appendChild(bubble);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function appendErrorMessage(err) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "chat-msg sage";

        const header = document.createElement("div");
        header.className = "msg-header";
        header.style.color = "var(--accent-red)";
        header.textContent = "SAGE Error";
        msgDiv.appendChild(header);

        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        bubble.style.borderColor = "var(--accent-red)";
        bubble.innerHTML = `<strong>Error:</strong> <pre>${escapeHtml(err)}</pre>`;

        msgDiv.appendChild(bubble);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function updateTelemetryAndTrace(telemetry, trace) {
        if (telemetry) {
            telemetryCard.style.display = "block";
            metricAgentCalls.textContent = telemetry.agent_calls || 0;
            metricDocCalls.textContent = telemetry.document_calls || 0;
            metricCoderCalls.textContent = telemetry.coder_calls || 0;
            metricSwitches.textContent = telemetry.model_switches || 0;
            metricWallTime.textContent = `${telemetry.total_wall_time.toFixed(2)}s`;
        }

        if (trace && trace.length > 0) {
            traceTimeline.innerHTML = "";
            trace.forEach((evt) => {
                const evtDiv = document.createElement("div");
                evtDiv.className = `trace-event ${evt.actor}`;

                const actorHeader = document.createElement("div");
                actorHeader.className = "trace-event-actor";
                const actorName = evt.actor === "gemma" ? "🧠 Gemma 4B" :
                                  evt.actor === "document_analyzer" ? "📄 Qwen3-VL" :
                                  evt.actor === "coder" ? "💻 Qwen2.5-Coder" :
                                  evt.actor === "sandbox" ? "📦 Docker Sandbox" : evt.actor;
                
                const timeInfo = evt.duration ? `(${evt.duration.toFixed(1)}s)` :
                                 evt.wall_time_ms ? `(${evt.wall_time_ms.toFixed(0)}ms)` : "";
                actorHeader.innerHTML = `<span>${actorName}</span> <span style="font-weight:normal;color:var(--text-muted)">${timeInfo}</span>`;
                evtDiv.appendChild(actorHeader);

                const detail = document.createElement("div");
                detail.className = "trace-event-detail";
                if (evt.action === "requested_tools") {
                    detail.textContent = `Requested: ${evt.calls.map(c => c.tool).join(", ")}`;
                } else if (evt.action === "code_generated") {
                    detail.textContent = `Generated Code: ${evt.snippet}`;
                } else if (evt.action === "analyzed_content") {
                    detail.textContent = `Analyzed: ${evt.input}\nResult: ${evt.snippet}`;
                } else if (evt.action === "executed_code") {
                    const statusTag = evt.status ? evt.status.toUpperCase() : "DONE";
                    const stdoutSnippet = evt.stdout_preview ? `\nStdout: ${evt.stdout_preview}` : "";
                    detail.textContent = `[${statusTag}] Exit ${evt.exit_code} | Memory: ${evt.memory_peak_mb}MB | Attempts: ${evt.attempts}${stdoutSnippet}`;
                } else if (evt.action === "final_synthesis") {
                    detail.textContent = `Synthesized final answer for user`;
                } else {
                    detail.textContent = JSON.stringify(evt);
                }
                evtDiv.appendChild(detail);

                traceTimeline.appendChild(evtDiv);
            });
        }
    }

    function escapeHtml(text) {
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function formatMarkdown(text) {
        if (!text) return "";
        let formatted = escapeHtml(text);

        // Code blocks
        formatted = formatted.replace(/```([a-zA-Z0-9_]*)\n([\s\S]*?)```/g, (match, lang, code) => {
            return `<pre><code class="language-${lang}">${code}</code></pre>`;
        });

        // Inline code
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Headers
        formatted = formatted.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        formatted = formatted.replace(/^## (.*$)/gim, '<h2>$1</h2>');
        formatted = formatted.replace(/^# (.*$)/gim, '<h1>$1</h1>');

        // Line breaks
        formatted = formatted.replace(/\n/g, '<br>');

        // Fix pre tags breaks
        formatted = formatted.replace(/<pre><code.*?>[\s\S]*?<\/code><\/pre>/g, (match) => {
            return match.replace(/<br>/g, '\n');
        });

        return formatted;
    }
});
