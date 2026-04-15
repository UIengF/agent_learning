const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const indexInput = document.querySelector("#index-dir");
const sessionInput = document.querySelector("#session-id");
const resumeInput = document.querySelector("#resume");
const submitButton = document.querySelector("#submit-button");
const clearButton = document.querySelector("#clear-button");
const statusText = document.querySelector("#status-text");
const history = document.querySelector("#history");

let hasAnswers = false;
const conversationHistory = [];

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
}

function clearEmptyState() {
  if (!hasAnswers) {
    history.innerHTML = "";
    hasAnswers = true;
  }
}

function addAnswerCard(data) {
  clearEmptyState();
  const card = document.createElement("article");
  card.className = "answer-card";
  card.innerHTML = `
    <h2>${escapeHtml(data.question)}</h2>
    <p>回答</p>
    <div class="markdown-answer">${renderMarkdown(data.answer)}</div>
    <div class="meta">
      <span>会话：${escapeHtml(data.session_id)}</span>
      <span>索引：${escapeHtml(data.index_dir)}</span>
      <span>耗时：${escapeHtml(String(data.elapsed_seconds))} 秒</span>
      <span>${data.resume ? "继续会话" : "新问题"}</span>
    </div>
  `;
  history.prepend(card);
  conversationHistory.push({question: data.question, answer: data.answer});
  if (conversationHistory.length > 8) {
    conversationHistory.shift();
  }
}

function addErrorCard(question, message) {
  clearEmptyState();
  const card = document.createElement("article");
  card.className = "answer-card";
  card.innerHTML = `
    <h2>${escapeHtml(question || "请求失败")}</h2>
    <p class="error">${escapeHtml(message)}</p>
  `;
  history.prepend(card);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
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

  for (const line of lines) {
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
    setStatus("请输入问题。", true);
    questionInput.focus();
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "分析中...";
  setStatus("正在检索和整理资料。");

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "请求失败。");
    }
    addAnswerCard(data);
    setStatus("回答已生成。");
  } catch (error) {
    const message = error instanceof Error ? error.message : "请求失败。";
    addErrorCard(payload.question, message);
    setStatus(message, true);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "开始回答";
  }
});

clearButton.addEventListener("click", () => {
  hasAnswers = false;
  conversationHistory.length = 0;
  history.innerHTML = `
    <article class="empty-state">
      <h2>先提一个问题</h2>
      <p>回答会显示在这里，并保留本页最近几轮记录。</p>
    </article>
  `;
  setStatus("准备就绪");
});

loadConfig().catch(() => {});
