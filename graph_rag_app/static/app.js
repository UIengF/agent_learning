const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const indexInput = document.querySelector("#index-dir");
const sessionInput = document.querySelector("#session-id");
const resumeInput = document.querySelector("#resume");
const submitButton = document.querySelector("#submit-button");
const clearButton = document.querySelector("#clear-button");
const statusText = document.querySelector("#status-text");
const statusIndicator = document.querySelector("#status-indicator");
const history = document.querySelector("#history");
const questionPane = document.querySelector("#question-pane");
const desktopSidebarToggle = document.querySelector("#desktop-sidebar-toggle");
const mobileSidebarToggle = document.querySelector("#mobile-sidebar-toggle");

let hasAnswers = false;
let isSidebarOpen = window.innerWidth > 860;
const conversationHistory = [];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setStatus(text, isError = false, isLoading = false) {
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
  statusIndicator.classList.toggle("loading", isLoading);
}

function renderEmptyState() {
  return `
    <article class="empty-state">
      <div class="empty-icon">⌕</div>
      <h2>先提一个问题</h2>
      <p>回答会显示在这里，并保留本页最近几轮记录。</p>
    </article>
  `;
}

function clearEmptyState() {
  if (!hasAnswers) {
    history.innerHTML = "";
    hasAnswers = true;
  }
}

function setSidebarState(open) {
  isSidebarOpen = open;
  questionPane.classList.toggle("open", open);
  mobileSidebarToggle.classList.toggle("active", open);
  if (desktopSidebarToggle) {
    desktopSidebarToggle.classList.toggle("hidden", open);
  }
}

function syncResponsiveSidebar() {
  if (window.innerWidth > 860) {
    setSidebarState(true);
  } else {
    setSidebarState(false);
  }
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function splitTableRow(line) {
  let normalized = String(line ?? "").trim();
  if (normalized.startsWith("|")) {
    normalized = normalized.slice(1);
  }
  if (normalized.endsWith("|")) {
    normalized = normalized.slice(0, -1);
  }
  return normalized.split("|").map((cell) => cell.trim());
}

function isTableSeparatorLine(line) {
  const cells = splitTableRow(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function looksLikeTableBlock(lines, index) {
  if (index + 1 >= lines.length) {
    return false;
  }
  const header = String(lines[index] ?? "").trim();
  const separator = String(lines[index + 1] ?? "").trim();
  if (!header.includes("|") || !separator.includes("|")) {
    return false;
  }
  const headerCells = splitTableRow(header);
  const separatorCells = splitTableRow(separator);
  return headerCells.length > 1 && headerCells.length === separatorCells.length && isTableSeparatorLine(separator);
}

function renderTable(headerLine, bodyLines) {
  const headers = splitTableRow(headerLine);
  const rows = bodyLines
    .map((line) => splitTableRow(line))
    .filter((cells) => cells.length > 0)
    .map((cells) => {
      const padded = [...cells];
      while (padded.length < headers.length) {
        padded.push("");
      }
      return padded.slice(0, headers.length);
    });

  return `
    <div class="table-wrap">
      <table class="markdown-table">
        <thead>
          <tr>${headers.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${rows.map((cells) => `<tr>${cells.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderMarkdown(markdown) {
  const lines = String(markdown ?? "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let inCodeBlock = false;
  let codeLines = [];
  let inList = false;
  let listType = "";
  let paragraph = [];

  function closeParagraph() {
    if (paragraph.length > 0) {
      html.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  }

  function closeList() {
    if (inList) {
      html.push(`</${listType}>`);
      inList = false;
      listType = "";
    }
  }

  function openList(type) {
    if (!inList || listType !== type) {
      closeParagraph();
      closeList();
      listType = type;
      html.push(`<${listType}>`);
      inList = true;
    }
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      if (inCodeBlock) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
        inCodeBlock = false;
      } else {
        closeParagraph();
        closeList();
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    if (!trimmed) {
      closeParagraph();
      closeList();
      continue;
    }

    if (looksLikeTableBlock(lines, index)) {
      closeParagraph();
      closeList();
      const bodyLines = [];
      let bodyIndex = index + 2;
      while (bodyIndex < lines.length) {
        const candidate = String(lines[bodyIndex] ?? "");
        const candidateTrimmed = candidate.trim();
        if (!candidateTrimmed || !candidateTrimmed.includes("|")) {
          break;
        }
        bodyLines.push(candidateTrimmed);
        bodyIndex += 1;
      }
      html.push(renderTable(trimmed, bodyLines));
      index = bodyIndex - 1;
      continue;
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      closeParagraph();
      closeList();
      const level = heading[1].length + 2;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const unordered = trimmed.match(/^[-*]\s+(.+)$/);
    if (unordered) {
      openList("ul");
      html.push(`<li>${renderInlineMarkdown(unordered[1])}</li>`);
      continue;
    }

    const ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (ordered) {
      openList("ol");
      html.push(`<li>${renderInlineMarkdown(ordered[1])}</li>`);
      continue;
    }

    if (trimmed.startsWith(">")) {
      closeParagraph();
      closeList();
      html.push(`<blockquote>${renderInlineMarkdown(trimmed.replace(/^>\s?/, ""))}</blockquote>`);
      continue;
    }

    closeList();
    paragraph.push(trimmed);
  }

  if (inCodeBlock) {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  }
  closeParagraph();
  closeList();
  return html.join("");
}

function addAnswerCard(data) {
  clearEmptyState();
  const card = document.createElement("article");
  card.className = "answer-card";
  card.innerHTML = `
    <div class="answer-card-header">
      <div class="answer-card-title">
        <div class="answer-icon">✦</div>
        <h2>${escapeHtml(data.question)}</h2>
      </div>
      <span class="answer-time">${escapeHtml(String(data.elapsed_seconds))} s</span>
    </div>
    <div class="answer-card-body">
      <div class="markdown-answer">${renderMarkdown(data.answer)}</div>
      <div class="meta">
        <div class="meta-item">
          <span class="meta-icon">◫</span>
          <span>Index: ${escapeHtml(data.index_dir)}</span>
        </div>
        <div class="meta-item">
          <span class="meta-icon">◎</span>
          <span>Session: ${escapeHtml(data.session_id)}</span>
        </div>
        <div class="meta-item">
          <span class="meta-icon">↺</span>
          <span>${data.resume ? "Continue" : "Fresh"}</span>
        </div>
      </div>
    </div>
  `;
  history.prepend(card);
  conversationHistory.push({ question: data.question, answer: data.answer });
  if (conversationHistory.length > 8) {
    conversationHistory.shift();
  }
}

function addErrorCard(question, message) {
  clearEmptyState();
  const card = document.createElement("article");
  card.className = "answer-card";
  card.innerHTML = `
    <div class="answer-card-header">
      <div class="answer-card-title">
        <div class="answer-icon">!</div>
        <h2>${escapeHtml(question || "请求失败")}</h2>
      </div>
    </div>
    <div class="answer-card-body">
      <p class="error">${escapeHtml(message)}</p>
    </div>
  `;
  history.prepend(card);
}

async function loadConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) {
    return;
  }
  const config = await response.json();
  indexInput.value = config.index_dir || indexInput.value;
  sessionInput.value = config.session_id || sessionInput.value;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    question: questionInput.value.trim(),
    index_dir: indexInput.value.trim(),
    session_id: sessionInput.value.trim(),
    resume: resumeInput.checked,
    history: resumeInput.checked ? conversationHistory.slice(-4) : [],
  };

  if (!payload.question) {
    setStatus("请输入问题。", true, false);
    questionInput.focus();
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "分析中...";
  setStatus("正在检索和整理资料。", false, true);

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error?.message || data.error || "请求失败。");
    }
    addAnswerCard(data);
    setStatus("回答已生成。", false, false);
    if (window.innerWidth <= 860) {
      setSidebarState(false);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "请求失败。";
    addErrorCard(payload.question, message);
    setStatus(message, true, false);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "开始回答";
  }
});

clearButton.addEventListener("click", () => {
  hasAnswers = false;
  conversationHistory.length = 0;
  history.innerHTML = renderEmptyState();
  setStatus("准备就绪", false, false);
});

desktopSidebarToggle?.addEventListener("click", () => setSidebarState(true));
mobileSidebarToggle?.addEventListener("click", () => setSidebarState(!isSidebarOpen));
window.addEventListener("resize", syncResponsiveSidebar);

history.innerHTML = renderEmptyState();
syncResponsiveSidebar();
loadConfig().catch(() => {});
