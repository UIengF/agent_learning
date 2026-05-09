const sessionsEl = document.querySelector("#sessions");
const messagesEl = document.querySelector("#messages");
const summaryEl = document.querySelector("#summary");
const timelineEl = document.querySelector("#timeline");
const detailEl = document.querySelector("#detail");
const refreshButton = document.querySelector("#refresh");
const newSessionButton = document.querySelector("#new-session");
const runButton = document.querySelector("#run-agent");
const runStatusEl = document.querySelector("#run-status");
const taskInput = document.querySelector("#run-task");
const sessionInput = document.querySelector("#run-session");
const workspaceInput = document.querySelector("#run-workspace");
const modeInput = document.querySelector("#run-mode");
const filters = Array.from(document.querySelectorAll(".filters input"));

let currentSession = null;
let currentEvents = [];
let selectedEventIndex = -1;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noreferrer">$1</a>',
    );
}

function markdownToHtml(value) {
  const lines = String(value ?? "").replaceAll("\r\n", "\n").split("\n");
  const html = [];
  let inCode = false;
  let codeLines = [];
  let listType = null;

  function closeList(nextType = null) {
    if (listType && listType !== nextType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  }

  function closeCode() {
    if (!inCode) return;
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    codeLines = [];
    inCode = false;
  }

  lines.forEach((line) => {
    if (line.trim().startsWith("```")) {
      if (inCode) closeCode();
      else {
        closeList();
        inCode = true;
        codeLines = [];
      }
      return;
    }
    if (inCode) {
      codeLines.push(line);
      return;
    }
    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    if (heading) {
      closeList();
      const level = heading[1].length + 2;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      return;
    }
    const unorderedItem = /^\s*[-*]\s+(.+)$/.exec(line);
    const orderedItem = /^\s*\d+[.)]\s+(.+)$/.exec(line);
    if (unorderedItem || orderedItem) {
      const nextType = unorderedItem ? "ul" : "ol";
      closeList(nextType);
      if (!listType) {
        html.push(`<${nextType}>`);
        listType = nextType;
      }
      html.push(`<li>${renderInlineMarkdown((unorderedItem || orderedItem)[1])}</li>`);
      return;
    }
    if (!line.trim()) {
      closeList();
      return;
    }
    closeList();
    html.push(`<p>${renderInlineMarkdown(line)}</p>`);
  });

  closeCode();
  closeList();
  return html.join("");
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || payload.error || response.statusText);
  return payload;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || data.error || response.statusText);
  return data;
}

async function loadChatSessions() {
  sessionsEl.innerHTML = '<p class="meta">Loading conversations...</p>';
  const payload = await fetchJson("/api/chat/sessions");
  const sessions = payload.sessions || [];
  if (!sessions.length) {
    sessionsEl.innerHTML = '<p class="meta">No conversations yet.</p>';
    return;
  }
  sessionsEl.innerHTML = sessions
    .map((session) => `
      <button class="session" data-session="${escapeHtml(session.session_id)}" type="button">
        <strong>${escapeHtml(session.session_id)}</strong>
        <span>${escapeHtml(session.message_count)} messages</span>
        <small>${escapeHtml(session.latest_message || session.workspace || "")}</small>
      </button>
    `)
    .join("");
  sessionsEl.querySelectorAll(".session").forEach((node) => {
    node.addEventListener("click", () => selectChatSession(node.dataset.session));
  });
  const params = new URLSearchParams(window.location.search);
  const initialSession = params.get("session");
  if (initialSession && initialSession !== currentSession) {
    await selectChatSession(initialSession);
  }
}

async function selectChatSession(sessionId) {
  currentSession = sessionId;
  sessionInput.value = sessionId;
  sessionsEl.querySelectorAll(".session").forEach((node) => {
    node.classList.toggle("active", node.dataset.session === sessionId);
  });
  const session = await fetchJson(`/api/chat/sessions/${encodeURIComponent(sessionId)}`);
  workspaceInput.value = session.workspace || workspaceInput.value;
  renderMessages(session.history || []);
  await selectTraceSession(sessionId);
  window.history.replaceState(null, "", `/?session=${encodeURIComponent(sessionId)}`);
}

function renderMessages(history) {
  if (!history.length) {
    messagesEl.innerHTML = `
      <div class="empty-state">
        <h2>Start a coding conversation</h2>
        <p>Send a request, then keep using the same session for follow-up changes.</p>
      </div>
    `;
    return;
  }
  messagesEl.innerHTML = history
    .map((message) => `
      <article class="message ${escapeHtml(message.role)}">
        <div class="message-role">${escapeHtml(message.role)}</div>
        <div class="message-body markdown-body">${markdownToHtml(message.content || "")}</div>
      </article>
    `)
    .join("");
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendPendingMessage(content) {
  const existing = messagesEl.querySelector(".empty-state");
  if (existing) messagesEl.innerHTML = "";
  messagesEl.insertAdjacentHTML(
    "beforeend",
    `
      <article class="message user">
        <div class="message-role">user</div>
        <div class="message-body markdown-body">${markdownToHtml(content)}</div>
      </article>
      <article class="message assistant pending">
        <div class="message-role">assistant</div>
        <div class="message-body markdown-body"><p>Running...</p></div>
      </article>
    `,
  );
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function runAgent() {
  const task = taskInput.value.trim();
  if (!task) return;
  const payload = {
    mode: modeInput.value,
    workspace: workspaceInput.value,
    session_id: sessionInput.value.trim(),
    task,
  };
  runButton.disabled = true;
  taskInput.disabled = true;
  runStatusEl.textContent = "Running...";
  appendPendingMessage(task);
  try {
    const result = await postJson("/api/run", payload);
    currentSession = result.session_id;
    sessionInput.value = result.session_id;
    taskInput.value = "";
    runStatusEl.textContent = `Completed: ${result.session_id}`;
    await loadChatSessions();
    await selectChatSession(result.session_id);
  } catch (error) {
    runStatusEl.textContent = "Run failed";
    const pending = messagesEl.querySelector(".message.pending .message-body");
    if (pending) pending.innerHTML = markdownToHtml(error.message);
  } finally {
    runButton.disabled = false;
    taskInput.disabled = false;
    taskInput.focus();
  }
}

async function selectTraceSession(sessionId) {
  selectedEventIndex = -1;
  try {
    const [eventsPayload, summary] = await Promise.all([
      fetchJson(`/api/sessions/${encodeURIComponent(sessionId)}`),
      fetchJson(`/api/sessions/${encodeURIComponent(sessionId)}/summary`),
    ]);
    currentEvents = eventsPayload.events || [];
    renderSummary(summary);
  } catch (error) {
    currentEvents = [];
    summaryEl.innerHTML = `
      <h2>Session activity</h2>
      <p class="meta">No trace events for this conversation yet.</p>
    `;
  }
  renderTimeline();
  renderDetail(null);
}

function renderSummary(summary) {
  summaryEl.innerHTML = `
    <h2>${escapeHtml(summary.session_id)}</h2>
    <p class="meta">${escapeHtml(summary.trace_file)}</p>
    <div class="summary-grid">
      <div class="metric"><span>Events</span><strong>${summary.event_count}</strong></div>
      <div class="metric"><span>Model</span><strong>${summary.model_call_count}</strong></div>
      <div class="metric"><span>Tools</span><strong>${summary.tool_call_count}</strong></div>
    </div>
    <div class="section">
      <h3>Changed files</h3>
      <p>${escapeHtml((summary.changed_files || []).join(", ") || "none")}</p>
    </div>
    <div class="section">
      <h3>Validation</h3>
      <pre>${escapeHtml((summary.validation_commands || []).map((item) => `${item.status}: ${item.command}`).join("\n") || "none")}</pre>
    </div>
  `;
}

function allowedEventTypes() {
  return new Set(filters.filter((node) => node.checked).map((node) => node.value));
}

function renderTimeline() {
  const allowed = allowedEventTypes();
  const rows = currentEvents
    .map((event, index) => ({ event, index }))
    .filter(({ event }) => allowed.has(event.event_type));
  timelineEl.innerHTML = rows
    .map(({ event, index }) => `
      <button class="event ${index === selectedEventIndex ? "active" : ""}" data-index="${index}" type="button">
        <strong>${escapeHtml(event.event_type)}${event.tool_name ? ` / ${escapeHtml(event.tool_name)}` : ""}</strong>
        <span class="badge ${escapeHtml(event.status || "ok")}">${escapeHtml(event.status || "ok")}</span>
        <small>${escapeHtml(event.input_summary || "")}</small>
      </button>
    `)
    .join("") || '<p class="meta">No trace events match the filters.</p>';
  timelineEl.querySelectorAll(".event").forEach((node) => {
    node.addEventListener("click", () => {
      selectedEventIndex = Number(node.dataset.index);
      renderTimeline();
      renderDetail(currentEvents[selectedEventIndex]);
    });
  });
}

function renderDetail(event) {
  if (!event) {
    detailEl.innerHTML = "<h2>Event detail</h2><p class=\"meta\">Select an event.</p>";
    return;
  }
  detailEl.innerHTML = `
    <h2>${escapeHtml(event.event_type)}${event.tool_name ? ` / ${escapeHtml(event.tool_name)}` : ""}</h2>
    <p class="meta">${escapeHtml(event.timestamp || "")}</p>
    <div class="section"><h3>Status</h3><span class="badge ${escapeHtml(event.status || "ok")}">${escapeHtml(event.status || "ok")}</span></div>
    <div class="section"><h3>Input</h3><pre>${escapeHtml(event.input_summary || "")}</pre></div>
    <div class="section"><h3>Output</h3><div class="markdown-body">${markdownToHtml(event.output_summary || "")}</div></div>
  `;
}

refreshButton.addEventListener("click", loadChatSessions);
newSessionButton.addEventListener("click", () => {
  currentSession = null;
  sessionInput.value = "";
  taskInput.value = "";
  renderMessages([]);
  currentEvents = [];
  renderTimeline();
  renderDetail(null);
  summaryEl.innerHTML = "<h2>Session activity</h2><p class=\"meta\">Trace details appear after a run.</p>";
  window.history.replaceState(null, "", "/");
  taskInput.focus();
});
runButton.addEventListener("click", runAgent);
taskInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") runAgent();
});
filters.forEach((node) => node.addEventListener("change", renderTimeline));

loadChatSessions().catch((error) => {
  sessionsEl.innerHTML = `<p class="meta">Failed to load conversations: ${escapeHtml(error.message)}</p>`;
});
taskInput.focus();
