/* ============================================================
   Cravv Observatory — Frontend Application
   ============================================================ */

// ── State ────────────────────────────────────────────────────
let currentMode = 'debug';
let history = [];
let isStreaming = false;

// ── DOM References ───────────────────────────────────────────
const indexStatus    = document.getElementById('index-status');
const modelSelect    = document.getElementById('model-select');
const chatMessages   = document.getElementById('chat-messages');
const queryInput     = document.getElementById('query-input');
const sendBtn        = document.getElementById('send-btn');
const retrievalStats = document.getElementById('retrieval-stats');
const sourceViewer   = document.getElementById('source-viewer');
const resizeHandle   = document.getElementById('resize-handle');

// ── Mode placeholders ────────────────────────────────────────
const MODE_PLACEHOLDERS = {
  debug:   'Paste your error or stack trace...',
  feature: 'Describe what you want to build...',
  explain: 'What part of the codebase do you want to understand?',
};

// ── Mode Switching ───────────────────────────────────────────
document.querySelectorAll('.mode-tab').forEach(function (tab) {
  tab.addEventListener('click', function () {
    document.querySelectorAll('.mode-tab').forEach(function (t) {
      t.classList.remove('mode-tab--active');
      t.setAttribute('aria-selected', 'false');
    });
    tab.classList.add('mode-tab--active');
    tab.setAttribute('aria-selected', 'true');
    currentMode = tab.dataset.mode;
    queryInput.placeholder = MODE_PLACEHOLDERS[currentMode] || '';
  });
});

// ── Send Button and Keyboard Shortcut ────────────────────────
sendBtn.addEventListener('click', sendQuery);

queryInput.addEventListener('keydown', function (e) {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    sendQuery();
  }
});

// ── Send Query ───────────────────────────────────────────────
async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query || isStreaming) return;

  isStreaming = true;
  sendBtn.disabled = true;
  queryInput.value = '';

  addMessage('user', query);

  const thinkingEl = createThinkingIndicator();
  chatMessages.appendChild(thinkingEl);
  scrollToBottom();

  const assistantEl = createMessageElement('assistant');
  let rawText = '';

  try {
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        mode: currentMode,
        model: modelSelect.value,
        history: history.slice(-10),
      }),
    });

    if (!response.ok) {
      throw new Error('HTTP ' + response.status + ': ' + response.statusText);
    }

    thinkingEl.remove();
    chatMessages.appendChild(assistantEl);
    scrollToBottom();

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;

      buffer += decoder.decode(chunk.value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (let i = 0; i < lines.length; i++) {
        const trimmed = lines[i].trim();
        if (!trimmed.startsWith('data: ')) continue;

        const jsonStr = trimmed.slice(6);
        if (!jsonStr) continue;

        let event;
        try {
          event = JSON.parse(jsonStr);
        } catch (parseErr) {
          continue;
        }

        if (event.type === 'token') {
          rawText += event.content;
          assistantEl.textContent = rawText;
          scrollToBottom();
        } else if (event.type === 'done') {
          finalizeAssistantMessage(assistantEl, rawText);
          updateRetrievalStats(event.stats, event.sources);
          history.push({ role: 'user', content: query });
          history.push({ role: 'assistant', content: rawText });
          scrollToBottom();
        } else if (event.type === 'error') {
          assistantEl.classList.add('chat-message--error');
          assistantEl.textContent = 'Error: ' + event.content;
          scrollToBottom();
        }
      }
    }
  } catch (err) {
    thinkingEl.remove();
    if (!assistantEl.isConnected) chatMessages.appendChild(assistantEl);
    assistantEl.textContent = 'Connection error: ' + err.message;
    scrollToBottom();
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    queryInput.focus();
  }
}

// ── Finalize Assistant Message ───────────────────────────────
function finalizeAssistantMessage(el, rawText) {
  el.innerHTML = renderMarkdown(rawText);
  el.querySelectorAll('pre code').forEach(function (block) {
    hljs.highlightElement(block);
  });
  attachFileRefListeners(el);
}

// ── File Reference Detection ─────────────────────────────────
var FILE_PATH_PATTERN = /\blib\/[\w/.\\-]+\.dart\b/g;

function attachFileRefListeners(el) {
  // 1. Intercept standard markdown links created with file://
  el.querySelectorAll('a[href^="file://"]').forEach(function(a) {
    var path = a.getAttribute('href').replace('file://', '');
    if (path.startsWith('/')) {
        path = path.substring(1);
    }
    a.classList.add('file-ref');
    a.addEventListener('click', function(e) {
      e.preventDefault();
      loadSource(path);
    });
  });

  // 2. Fallback: Parse plain text for unformatted file paths
  var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
    acceptNode: function (node) {
      var parent = node.parentElement;
      while (parent && parent !== el) {
        if (parent.tagName === 'CODE' || parent.tagName === 'PRE') {
          return NodeFilter.FILTER_REJECT;
        }
        parent = parent.parentElement;
      }
      FILE_PATH_PATTERN.lastIndex = 0;
      return FILE_PATH_PATTERN.test(node.textContent)
        ? NodeFilter.FILTER_ACCEPT
        : NodeFilter.FILTER_SKIP;
    },
  });

  var textNodes = [];
  var node;
  while ((node = walker.nextNode())) {
    textNodes.push(node);
  }

  for (var n = 0; n < textNodes.length; n++) {
    var textNode = textNodes[n];
    FILE_PATH_PATTERN.lastIndex = 0;
    var text = textNode.textContent;
    var frag = document.createDocumentFragment();
    var lastIndex = 0;
    var match;

    while ((match = FILE_PATH_PATTERN.exec(text)) !== null) {
      if (match.index > lastIndex) {
        frag.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }
      var a = document.createElement('a');
      a.className = 'file-ref';
      a.dataset.path = match[0];
      a.textContent = match[0];
      a.href = '#';
      a.title = 'View source';
      (function (capturedPath) {
        a.addEventListener('click', function (e) {
          e.preventDefault();
          loadSource(capturedPath);
        });
      })(match[0]);
      frag.appendChild(a);
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < text.length) {
      frag.appendChild(document.createTextNode(text.slice(lastIndex)));
    }

    textNode.parentNode.replaceChild(frag, textNode);
  }
}

// ── Helper: Add Messages ─────────────────────────────────────
function addMessage(role, content) {
  var el = createMessageElement(role);
  if (role === 'user') {
    el.textContent = content;
  } else {
    el.innerHTML = renderMarkdown(content);
    attachFileRefListeners(el);
  }
  chatMessages.appendChild(el);
  scrollToBottom();
  return el;
}

function createMessageElement(role) {
  var el = document.createElement('div');
  el.className = 'chat-message chat-message--' + role;
  return el;
}

function createThinkingIndicator() {
  var el = document.createElement('div');
  el.className = 'chat-message chat-message--thinking';
  for (var i = 0; i < 3; i++) {
    var dot = document.createElement('span');
    dot.className = 'thinking-dot';
    el.appendChild(dot);
  }
  return el;
}

// ── Helper: Markdown Rendering ────────────────────────────────
function renderMarkdown(text) {
  return marked.parse(text, {
    breaks: true,
    gfm: true,
  });
}

// ── Helper: Retrieval Stats ───────────────────────────────────
function updateRetrievalStats(stats, sources) {
  if (!stats && !sources) {
    retrievalStats.textContent = 'Ready';
    return;
  }

  var chunks  = (stats && stats.chunks_retrieved != null) ? stats.chunks_retrieved : (sources ? sources.length : 0);
  var files   = countUniqueFiles(sources);
  var score   = (stats && stats.avg_score != null)  ? ('avg score ' + stats.avg_score.toFixed(2)) : null;
  var latency = (stats && stats.latency_ms != null) ? (stats.latency_ms + 'ms') : null;

  var parts = ['Retrieved: ' + chunks + (chunks !== 1 ? ' chunks' : ' chunk')];
  if (files > 0) parts.push('from ' + files + (files !== 1 ? ' files' : ' file'));
  if (score)     parts.push('(' + score + ')');
  if (latency)   parts.push('in ' + latency);

  retrievalStats.textContent = parts.join(' ');
}

function countUniqueFiles(sources) {
  if (!sources || !sources.length) return 0;
  var paths = new Set(
    sources.map(function (s) { return s.path || s.file || s.source || ''; }).filter(Boolean)
  );
  return paths.size;
}

// ── Helper: Auto-scroll ───────────────────────────────────────
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── Helper: HTML Escaping ─────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Source Viewer ─────────────────────────────────────────────
async function loadSource(path) {
  sourceViewer.textContent = 'Loading ' + path + '…';
  try {
    var response = await fetch('/api/sources?path=' + encodeURIComponent(path));
    if (!response.ok) throw new Error('HTTP ' + response.status);
    var data = await response.json();

    var lang = data.language || 'text';
    var highlighted;
    try {
      highlighted = hljs.highlight(data.content, { language: lang === 'dart' ? 'dart' : lang }).value;
    } catch (hlErr) {
      highlighted = escapeHtml(data.content);
    }

    // Build source viewer DOM safely
    sourceViewer.textContent = '';
    var header = document.createElement('div');
    header.className = 'source-header';
    header.textContent = path;
    sourceViewer.appendChild(header);

    var pre = document.createElement('pre');
    pre.className = 'source-code';
    var code = document.createElement('code');
    // highlighted is from hljs which produces safe HTML from our own file content
    // (local files only, path-validated by backend)
    code.innerHTML = formatLineNumbers(highlighted);
    pre.appendChild(code);
    sourceViewer.appendChild(pre);
  } catch (err) {
    sourceViewer.textContent = 'Could not load ' + path + ': ' + err.message;
  }
}

function formatLineNumbers(highlightedHtml) {
  var lines = highlightedHtml.split('\n');
  var rows = '';
  for (var i = 0; i < lines.length; i++) {
    var lineNum = i + 1;
    rows += '<div class="source-line">'
      + '<span class="source-line__num">' + lineNum + '</span>'
      + '<span class="source-line__code">' + (lines[i] || ' ') + '</span>'
      + '</div>';
  }
  return rows;
}

// ── Resize Handle ─────────────────────────────────────────────
(function () {
  if (!resizeHandle) return;
  var splitPane = resizeHandle.parentElement;
  var isDragging = false;
  var startX = 0;
  var startLeft = 0;

  resizeHandle.addEventListener('mousedown', function (e) {
    isDragging = true;
    startX = e.clientX;
    startLeft = splitPane.children[0].getBoundingClientRect().width;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', function (e) {
    if (!isDragging) return;
    var delta = e.clientX - startX;
    var newLeft = startLeft + delta;
    var totalWidth = splitPane.getBoundingClientRect().width;
    var minPanel = 200;
    newLeft = Math.max(minPanel, Math.min(newLeft, totalWidth - minPanel - 4));
    splitPane.style.gridTemplateColumns = newLeft + 'px 4px 1fr';
  });

  document.addEventListener('mouseup', function () {
    if (!isDragging) return;
    isDragging = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
})();

// ── Status Polling ───────────────────────────────────────────
async function pollStatus() {
  try {
    var response = await fetch('/api/status');
    if (!response.ok) throw new Error('HTTP ' + response.status);
    var data = await response.json();

    var filesIndexed = data.files_indexed || 0;
    var status       = data.index_status  || 'unknown';

    if (status === 'ready') {
      indexStatus.textContent = filesIndexed + ' files indexed';
      indexStatus.classList.remove('is-error');
      indexStatus.classList.add('is-ready');
    } else if (status === 'indexing') {
      indexStatus.textContent = 'Indexing\u2026 ' + filesIndexed + ' files';
      indexStatus.classList.remove('is-ready', 'is-error');
      setTimeout(pollStatus, 3000);
    } else {
      indexStatus.textContent = 'Status: ' + status;
      indexStatus.classList.remove('is-ready', 'is-error');
    }
  } catch (pollErr) {
    indexStatus.textContent = 'Index unavailable';
    indexStatus.classList.add('is-error');
    indexStatus.classList.remove('is-ready');
  }
}



// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function () {
  pollStatus();
  queryInput.focus();
});
