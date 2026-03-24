/* ============================================================
   Sourcefire — Frontend
   by Athar Wani
   ============================================================ */

// ── State ────────────────────────────────────────────────────
var currentMode = 'debug';
var chatHistory = [];
var isStreaming = false;

// ── DOM ──────────────────────────────────────────────────────
var chat          = document.getElementById('chat');
var queryInput    = document.getElementById('query-input');
var sendBtn       = document.getElementById('send-btn');
var indexStatus   = document.getElementById('index-status');
var langBadge     = document.getElementById('lang-badge');
var modelSelect   = document.getElementById('model-select');
var statsEl       = document.getElementById('retrieval-stats');
var sourceModal   = document.getElementById('source-modal');
var sourceViewer  = document.getElementById('source-viewer');
var sourceTitle   = document.getElementById('source-modal-title');
var sourceClose   = document.getElementById('source-modal-close');
var sourceBack    = sourceModal ? sourceModal.querySelector('.source-modal__backdrop') : null;

// ── Mode Switching ───────────────────────────────────────────
document.querySelectorAll('.mode-pill').forEach(function(pill) {
  pill.addEventListener('click', function() {
    document.querySelectorAll('.mode-pill').forEach(function(p) {
      p.classList.remove('mode-pill--active');
      p.setAttribute('aria-selected', 'false');
    });
    pill.classList.add('mode-pill--active');
    pill.setAttribute('aria-selected', 'true');
    currentMode = pill.dataset.mode;
  });
});

// ── Send ─────────────────────────────────────────────────────
sendBtn.addEventListener('click', sendQuery);
queryInput.addEventListener('keydown', function(e) {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); sendQuery(); }
});

// Auto-resize textarea
queryInput.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 200) + 'px';
});

// ── Send Query ───────────────────────────────────────────────
async function sendQuery() {
  var query = queryInput.value.trim();
  if (!query || isStreaming) return;

  isStreaming = true;
  sendBtn.disabled = true;
  queryInput.value = '';
  queryInput.style.height = 'auto';

  // Remove welcome
  var welcome = chat.querySelector('.welcome');
  if (welcome) welcome.remove();

  // User message
  appendMsg('user', query);

  // Assistant container with status timeline
  var container = document.createElement('div');
  container.className = 'msg msg--assistant';

  var timeline = document.createElement('div');
  timeline.className = 'status-timeline';
  container.appendChild(timeline);

  var thinkingEl = createThinking();
  container.appendChild(thinkingEl);

  var contentEl = document.createElement('div');
  contentEl.className = 'msg__content';
  contentEl.style.display = 'none';
  container.appendChild(contentEl);

  chat.appendChild(container);
  scrollToBottom();

  var rawText = '';

  try {
    var response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        mode: currentMode,
        model: modelSelect.value,
        history: chatHistory.slice(-10),
      }),
    });

    if (!response.ok) throw new Error('HTTP ' + response.status);

    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = '';

    while (true) {
      var chunk = await reader.read();
      if (chunk.done) break;

      buffer += decoder.decode(chunk.value, { stream: true });
      var lines = buffer.split('\n');
      buffer = lines.pop();

      for (var i = 0; i < lines.length; i++) {
        var trimmed = lines[i].trim();
        if (!trimmed.startsWith('data: ')) continue;
        var jsonStr = trimmed.slice(6);
        if (!jsonStr) continue;

        var event;
        try { event = JSON.parse(jsonStr); } catch(e) { continue; }

        if (event.type === 'status') {
          handleStatus(timeline, event);
        } else if (event.type === 'token') {
          if (thinkingEl.parentNode) thinkingEl.remove();
          contentEl.style.display = '';
          rawText += event.content;
          contentEl.textContent = rawText;
          scrollToBottom();
        } else if (event.type === 'done') {
          if (thinkingEl.parentNode) thinkingEl.remove();
          contentEl.style.display = '';
          addStatusTag(timeline, 'done', 'done');
          finalizeMsg(contentEl, rawText);
          updateStats(event.stats, event.sources);
          chatHistory.push({ role: 'user', content: query });
          chatHistory.push({ role: 'assistant', content: rawText });
          scrollToBottom();
        } else if (event.type === 'error') {
          if (thinkingEl.parentNode) thinkingEl.remove();
          contentEl.style.display = '';
          contentEl.textContent = 'Error: ' + event.content;
          addStatusTag(timeline, 'error', 'error');
          scrollToBottom();
        }
      }
    }
  } catch (err) {
    if (thinkingEl.parentNode) thinkingEl.remove();
    contentEl.style.display = '';
    contentEl.textContent = 'Connection error: ' + err.message;
    addStatusTag(timeline, 'error', 'failed');
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    queryInput.focus();
  }
}

// ── Status Handling ──────────────────────────────────────────
// One "live" tag that updates in-place, plus pinned tool/context tags.
function handleStatus(timeline, event) {
  var stage = event.stage;
  var live = timeline.querySelector('.status-tag--live');

  if (stage === 'retrieving') {
    setLiveTag(timeline, live, '\u26A1', 'retrieving...');
  } else if (stage === 'context_found') {
    // Replace live tag with a static context tag
    removeLive(timeline);
    pinTag(timeline, 'context', '\u25A0', event.chunks + ' chunks \u00B7 ' + event.files + ' files');
  } else if (stage === 'thinking') {
    setLiveTag(timeline, live, '\u26A1', 'thinking...');
  } else if (stage === 'generating') {
    setLiveTag(timeline, live, '\u26A1', 'generating...');
  } else if (stage === 'tool_call') {
    removeLive(timeline);
    pinTag(timeline, 'tool', '\u2699', event.tool);
  } else if (stage === 'tool_done') {
    // tool tag already pinned, nothing to do
  }
}

function setLiveTag(timeline, existing, icon, label) {
  if (existing) {
    // Update in-place
    existing.lastChild.textContent = label;
    existing.querySelector('.status-tag__icon').textContent = icon;
  } else {
    var tag = createTag('active', icon, label);
    tag.classList.add('status-tag--live');
    timeline.appendChild(tag);
  }
  scrollToBottom();
}

function removeLive(timeline) {
  var live = timeline.querySelector('.status-tag--live');
  if (live) live.remove();
}

function pinTag(timeline, type, icon, label) {
  timeline.appendChild(createTag(type, icon, label));
  scrollToBottom();
}

function addStatusTag(timeline, type, label) {
  // Used for final done/error
  removeLive(timeline);
  var icons = { 'done': '\u2713', 'error': '\u2717' };
  timeline.appendChild(createTag(type, icons[type] || '\u25CF', label));
  scrollToBottom();
}

function createTag(type, iconText, label) {
  var tag = document.createElement('span');
  tag.className = 'status-tag';
  if (type === 'active') tag.className += ' status-tag--active';
  else if (type === 'done') tag.className += ' status-tag--done';
  else if (type === 'tool') tag.className += ' status-tag--tool';
  else if (type === 'context') tag.className += ' status-tag--context';
  else if (type === 'error') tag.className += ' status-tag--error';

  var icon = document.createElement('span');
  icon.className = 'status-tag__icon';
  icon.textContent = iconText;
  tag.appendChild(icon);
  tag.appendChild(document.createTextNode(label));
  return tag;
}

// ── Finalize Message ─────────────────────────────────────────
function finalizeMsg(el, rawText) {
  // Safe: innerHTML set from renderMarkdown which processes our own markdown text
  el.innerHTML = renderMarkdown(rawText);
  el.querySelectorAll('pre code').forEach(function(block) {
    hljs.highlightElement(block);
  });
  attachFileRefListeners(el);
}

// ── Markdown Rendering ───────────────────────────────────────
var _FILE_LINK_RE = /\[([^\]]+)\]\(file:\/\/([^)]+)\)/g;
var _PLACEHOLDER_RE = /\u200B\u200BFILEREF\u200B([^\u200B]+)\u200B([^\u200B]+)\u200B\u200B/g;

function renderMarkdown(text) {
  var preprocessed = text.replace(_FILE_LINK_RE, function(match, linkText, path) {
    if (path.startsWith('/')) path = path.substring(1);
    return '\u200B\u200BFILEREF\u200B' + path + '\u200B' + linkText + '\u200B\u200B';
  });
  var html = marked.parse(preprocessed, { breaks: true, gfm: true });
  html = html.replace(_PLACEHOLDER_RE, function(match, path, linkText) {
    return '<a class="file-ref" href="#" data-path="' + escapeHtml(path) + '" title="View source">' + escapeHtml(linkText) + '</a>';
  });
  return html;
}

// ── File Ref Listeners ───────────────────────────────────────
var FILE_PATH_PATTERN = /\b[\w][\w./\\-]*\/[\w./\\-]+\.(?:py|js|jsx|ts|tsx|dart|go|rs|java|kt|swift|rb|php|c|cpp|cc|cxx|h|hpp|hxx|hh|yaml|yml|json|toml|md|html|css|sql|sh|proto|graphql|cmake)\b|\b[\w-]+\.(?:py|js|ts|dart|go|rs|java|c|cpp|h|hpp|yaml|yml|json|toml|md)\b/g;

function attachFileRefListeners(el) {
  el.querySelectorAll('a').forEach(function(a) {
    var path = '';
    if (a.dataset.path) {
      path = a.dataset.path;
    } else {
      var href = a.getAttribute('href') || '';
      if (href.startsWith('file://')) {
        path = decodeURIComponent(href.replace('file://', ''));
        if (path.startsWith('/')) path = path.substring(1);
      } else if (href.match(/^[\w][\w./\\-]+\.(?:py|js|ts|dart|go|rs|java|c|cpp|h|hpp|yaml|yml|json|md)$/)) {
        path = href;
      }
    }
    if (path) {
      a.classList.add('file-ref');
      a.removeAttribute('target');
      (function(p) { a.addEventListener('click', function(e) { e.preventDefault(); loadSource(p); }); })(path);
    }
  });

  // Fallback: plain text paths
  var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
    acceptNode: function(node) {
      var parent = node.parentElement;
      while (parent && parent !== el) {
        if (parent.tagName === 'CODE' || parent.tagName === 'PRE' || parent.tagName === 'A') return NodeFilter.FILTER_REJECT;
        parent = parent.parentElement;
      }
      FILE_PATH_PATTERN.lastIndex = 0;
      return FILE_PATH_PATTERN.test(node.textContent) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
    },
  });
  var textNodes = []; var n;
  while ((n = walker.nextNode())) textNodes.push(n);

  for (var t = 0; t < textNodes.length; t++) {
    var textNode = textNodes[t];
    FILE_PATH_PATTERN.lastIndex = 0;
    var text = textNode.textContent;
    var frag = document.createDocumentFragment();
    var lastIdx = 0; var m;
    while ((m = FILE_PATH_PATTERN.exec(text)) !== null) {
      if (m.index > lastIdx) frag.appendChild(document.createTextNode(text.slice(lastIdx, m.index)));
      var a = document.createElement('a');
      a.className = 'file-ref'; a.dataset.path = m[0]; a.textContent = m[0]; a.href = '#'; a.title = 'View source';
      (function(p) { a.addEventListener('click', function(e) { e.preventDefault(); loadSource(p); }); })(m[0]);
      frag.appendChild(a);
      lastIdx = m.index + m[0].length;
    }
    if (lastIdx < text.length) frag.appendChild(document.createTextNode(text.slice(lastIdx)));
    textNode.parentNode.replaceChild(frag, textNode);
  }
}

// ── Helpers ──────────────────────────────────────────────────
function appendMsg(role, content) {
  var el = document.createElement('div');
  el.className = 'msg msg--' + role;
  el.textContent = content;
  chat.appendChild(el);
  scrollToBottom();
}

function createThinking() {
  var el = document.createElement('div');
  el.className = 'thinking';
  for (var i = 0; i < 3; i++) { var d = document.createElement('span'); d.className = 'thinking__dot'; el.appendChild(d); }
  return el;
}

function scrollToBottom() { chat.scrollTop = chat.scrollHeight; }

function escapeHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function updateStats(stats, sources) {
  if (!stats && !sources) { statsEl.textContent = ''; return; }
  var c = sources ? sources.length : 0;
  var used = stats ? stats.chunks_used : c;
  statsEl.textContent = used + ' chunks \u00B7 ' + (stats ? stats.total_estimated : '?') + ' tokens est.';
}

// ── Source Viewer ────────────────────────────────────────────
function openModal() { if (sourceModal) sourceModal.hidden = false; }
function closeModal() { if (sourceModal) sourceModal.hidden = true; }

if (sourceClose) sourceClose.addEventListener('click', closeModal);
if (sourceBack) sourceBack.addEventListener('click', closeModal);
document.addEventListener('keydown', function(e) { if (e.key === 'Escape') closeModal(); });

async function loadSource(path) {
  if (!sourceViewer || !sourceModal) return;
  sourceTitle.textContent = path;
  sourceViewer.textContent = 'Loading\u2026';
  openModal();
  try {
    var response = await fetch('/api/sources?path=' + encodeURIComponent(path));
    if (!response.ok) throw new Error('HTTP ' + response.status);
    var data = await response.json();
    var lang = data.language || 'text';
    var highlighted;
    try { highlighted = hljs.highlight(data.content, { language: lang }).value; }
    catch(e) { try { highlighted = hljs.highlightAuto(data.content).value; } catch(e2) { highlighted = escapeHtml(data.content); } }
    sourceViewer.textContent = '';
    var pre = document.createElement('pre');
    var code = document.createElement('code');
    // Safe: highlighted is from hljs processing backend-validated local file content
    code.innerHTML = formatLines(highlighted);
    pre.appendChild(code);
    sourceViewer.appendChild(pre);
  } catch(err) {
    sourceViewer.textContent = 'Could not load ' + path + ': ' + err.message;
  }
}

function formatLines(html) {
  var lines = html.split('\n');
  var out = '';
  for (var i = 0; i < lines.length; i++) {
    out += '<div class="source-line"><span class="source-line__num">' + (i+1) + '</span><span class="source-line__code">' + (lines[i]||' ') + '</span></div>';
  }
  return out;
}

// ── Status Polling ───────────────────────────────────────────
async function pollStatus() {
  try {
    var res = await fetch('/api/status');
    if (!res.ok) throw new Error();
    var data = await res.json();
    if (data.index_status === 'ready') {
      indexStatus.textContent = data.files_indexed + ' files';
      indexStatus.classList.add('is-ready');
    } else {
      indexStatus.textContent = 'indexing';
      setTimeout(pollStatus, 3000);
    }
    if (data.language && data.language !== 'generic') {
      langBadge.textContent = data.language;
    }
    if (data.project_name) {
      document.title = data.project_name + ' — Sourcefire';
    }
  } catch(e) {
    indexStatus.textContent = 'offline';
    indexStatus.classList.add('is-error');
  }
}

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  pollStatus();
  queryInput.focus();
});
