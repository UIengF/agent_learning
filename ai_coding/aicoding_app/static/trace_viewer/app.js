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
const sessionSearchInput = document.querySelector("#session-search");
const filters = Array.from(document.querySelectorAll(".filters input"));

let currentSession = null;
let currentEvents = [];
let selectedEventIndex = -1;
let cachedSessions = [];

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

function fmtTime(ts) {
  if (!ts) return "";
  try {
    const date = new Date(ts);
    return Number.isNaN(date.getTime()) ? ts : date.toLocaleTimeString();
  } catch {
    return ts;
  }
}

function sessionStatus(session) {
  const latest = String(session.latest_message || "").toLowerCase();
  if (latest.includes("verified") || latest.includes("validation")) return "Verified";
  if (latest.includes("completed") || latest.includes("finished") || latest.includes("done")) return "Completed";
  if (latest.includes("modified") || latest.includes("changed") || latest.includes("updated")) return "Modified";
  return "Active";
}

function renderSessionList(sessions) {
  const query = (sessionSearchInput?.value || "").trim().toLowerCase();
  const visible = sessions.filter((session) => {
    const haystack = `${session.session_id || ""} ${session.latest_message || ""}`.toLowerCase();
    return !query || haystack.includes(query);
  });
  if (!visible.length) {
    sessionsEl.innerHTML = `<p class="meta empty-list">${query ? "No matching conversations." : "No conversations yet."}</p>`;
    return;
  }
  sessionsEl.innerHTML = visible
    .map((session) => {
      const status = sessionStatus(session);
      const updated = session.updated_at || session.latest_timestamp || session.created_at || "";
      const updatedLabel = updated ? `Updated ${fmtTime(updated)}` : "Updated recently";
      return `
        <button class="session ${session.session_id === currentSession ? "active" : ""}" data-session="${escapeHtml(session.session_id)}" type="button">
          <span class="session-dot" aria-hidden="true"></span>
          <strong>${escapeHtml(session.session_id)}</strong>
          <span class="session-count">${escapeHtml(session.message_count)} messages</span>
          <span class="session-badge ${escapeHtml(status.toLowerCase())}">${escapeHtml(status)}</span>
          <small>${escapeHtml(updatedLabel)}</small>
        </button>
      `;
    })
    .join("");
  sessionsEl.querySelectorAll(".session").forEach((node) => {
    node.addEventListener("click", () => selectChatSession(node.dataset.session));
  });
}

async function loadChatSessions() {
  sessionsEl.innerHTML = '<p class="meta">Loading conversations...</p>';
  const payload = await fetchJson("/api/chat/sessions");
  const sessions = payload.sessions || [];
  cachedSessions = sessions;
  renderSessionList(cachedSessions);
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
        <div class="welcome-icon" aria-hidden="true">&lt;/&gt;</div>
        <h2>Start a coding conversation</h2>
        <p>Describe a task, then follow up with changes in the same session.</p>
        <div class="quick-grid" aria-label="quick starts">
          <button type="button" class="quick-card">
            <span class="quick-icon">Code</span>
            <strong>Modify existing code</strong>
            <small>Update or refactor code in your project</small>
          </button>
          <button type="button" class="quick-card">
            <span class="quick-icon purple">New</span>
            <strong>Create feature</strong>
            <small>Build a new feature from scratch</small>
          </button>
          <button type="button" class="quick-card">
            <span class="quick-icon red">Fix</span>
            <strong>Debug issue</strong>
            <small>Investigate and fix bugs or errors</small>
          </button>
          <button type="button" class="quick-card">
            <span class="quick-icon green">QA</span>
            <strong>Review project</strong>
            <small>Analyze code quality and suggest improvements</small>
          </button>
        </div>
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
    <h2>Session activity</h2>
    <p class="meta">Trace details appear after a run.</p>
    <div class="summary-grid">
      <div class="metric"><span>Events</span><strong>${summary.event_count}</strong></div>
      <div class="metric"><span>Model</span><strong>${summary.model_call_count}</strong></div>
      <div class="metric"><span>Tools</span><strong>${summary.tool_call_count}</strong></div>
    </div>
    <p class="activity-session">${escapeHtml(summary.session_id)}</p>
  `;
}

function allowedEventTypes() {
  return new Set(filters.filter((node) => node.checked).map((node) => node.value));
}

function emptyTimelineHtml() {
  return `
    <button class="event placeholder" type="button" disabled>
      <span class="event-icon" aria-hidden="true"></span>
      <strong>Run started</strong>
      <time>10:24 AM</time>
      <small>Agent initialized</small>
    </button>
    <button class="event placeholder" type="button" disabled>
      <span class="event-icon" aria-hidden="true"></span>
      <strong>Tool used</strong>
      <time>10:24 AM</time>
      <small>read_file</small>
    </button>
    <button class="event placeholder" type="button" disabled>
      <span class="event-icon" aria-hidden="true"></span>
      <strong>Model response</strong>
      <time>10:24 AM</time>
      <small>Generated code changes</small>
    </button>
    <button class="event placeholder" type="button" disabled>
      <span class="event-icon" aria-hidden="true"></span>
      <strong>Run completed</strong>
      <time>10:24 AM</time>
      <small>All steps finished</small>
    </button>
  `;
}

function renderTimeline() {
  const allowed = allowedEventTypes();
  const rows = currentEvents
    .map((event, index) => ({ event, index }))
    .filter(({ event }) => allowed.has(event.event_type));
  if (!rows.length) {
    timelineEl.innerHTML = emptyTimelineHtml();
    return;
  }
  timelineEl.innerHTML = rows
    .map(({ event, index }) => {
      const label = {
        state_transition: event.status === "completed" ? "Run completed" : "Run started",
        tool_call: "Tool used",
        model_call: "Model response",
        final_response: "Run completed",
        model_fallback: "Fallback used",
      }[event.event_type] || event.event_type;
      const description = event.tool_name || event.input_summary || event.output_summary || "Agent initialized";
      return `
        <button class="event ${index === selectedEventIndex ? "active" : ""}" data-index="${index}" type="button">
          <span class="event-icon" aria-hidden="true"></span>
          <strong>${escapeHtml(label)}</strong>
          <time>${escapeHtml(fmtTime(event.timestamp))}</time>
          <small>${escapeHtml(description).slice(0, 120)}</small>
        </button>
      `;
    })
    .join("");
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
    detailEl.innerHTML = `
      <h2>Event detail</h2>
      <p class="meta">Select an event from the timeline.</p>
      <div class="detail-empty">
        <div class="detail-empty-icon" aria-hidden="true">[]</div>
        <strong>No event selected</strong>
        <span>Choose an event to view more details here.</span>
      </div>
    `;
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
if (sessionSearchInput) {
  sessionSearchInput.addEventListener("input", () => renderSessionList(cachedSessions));
}

loadChatSessions().catch((error) => {
  sessionsEl.innerHTML = `<p class="meta">Failed to load conversations: ${escapeHtml(error.message)}</p>`;
});
taskInput.focus();
