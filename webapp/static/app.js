/* Auto Research — web edition UI (vanilla JS, no frameworks, no external resources). */
'use strict';

(function () {

// ===== Constants ============================================================

var EPOCH_OFFSET = 978307200; // seconds between the Apple (2001) and Unix (1970) epochs
var PROVIDERS = ['codex', 'claude', 'opencode', 'antigravity'];
var PROVIDER_TITLES = { codex: 'Codex', claude: 'Claude', opencode: 'OpenCode', antigravity: 'Antigravity' };
var INSTALL_URLS = {
  codex: 'https://developers.openai.com/codex/cli',
  claude: 'https://code.claude.com/docs/en/overview',
  opencode: 'https://opencode.ai/docs',
  antigravity: 'https://antigravity.google/docs/cli/reference'
};
var TOKEN_KEY = 'auto-research-token';
var PROVIDER_KEY = 'auto-research-provider';
var SHARE_KEY = 'auto-research-share';

// ===== Icons (inline SVG, currentColor) =====================================

var ICONS = {
  logo: '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="2.7"/><circle cx="18.4" cy="12" r="2.7"/><circle cx="5.6" cy="12" r="2.7"/><circle cx="15.2" cy="6.6" r="2.7"/><circle cx="8.8" cy="6.6" r="2.7"/><circle cx="15.2" cy="17.4" r="2.7"/><circle cx="8.8" cy="17.4" r="2.7"/></svg>',
  edit: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M13.5 4H5.9A1.9 1.9 0 0 0 4 5.9v12.2A1.9 1.9 0 0 0 5.9 20h12.2a1.9 1.9 0 0 0 1.9-1.9V10.5"/><path d="M17.4 3.9l2.7 2.7-7.2 7.2-3.4.7.7-3.4z"/></svg>',
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="10.5" cy="10.5" r="6.2"/><path d="M15.2 15.2 20.2 20.2"/></svg>',
  xcirclefill: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="currentColor"/><path d="M9.3 9.3l5.4 5.4M14.7 9.3l-5.4 5.4" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/></svg>',
  refresh: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 12a8 8 0 1 1-2.9-6.15"/><path d="M20.4 3.6v4.1h-4.1"/></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>',
  dotted: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="8" stroke-dasharray="0.01 3.6"/></svg>',
  folder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><path d="M3.5 7A1.7 1.7 0 0 1 5.2 5.3h4.1l2 2.4h7.5A1.7 1.7 0 0 1 20.5 9.4v7.9a1.7 1.7 0 0 1-1.7 1.7H5.2a1.7 1.7 0 0 1-1.7-1.7z"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7.2 4.6a1 1 0 0 1 1.5-.87l11 6.9a1 1 0 0 1 0 1.72l-11 6.9a1 1 0 0 1-1.5-.86z" fill="currentColor"/></svg>',
  waveform: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 10v4M8.3 6.5v11M12.6 3.5v17M16.9 7v10M21 10v4"/></svg>',
  bubble: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><path d="M4 8.2C4 6.3 5.7 4.8 7.7 4.8h8.6c2 0 3.7 1.5 3.7 3.4v3.3c0 1.9-1.7 3.4-3.7 3.4h-6.2l-4.4 3.4c-.4.3-.9 0-.9-.4v-3.2A3.6 3.6 0 0 1 4 11.5z"/></svg>',
  xmark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round"><path d="M6.2 6.2l11.6 11.6M17.8 6.2 6.2 17.8"/></svg>',
  archive: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6.2h16v4H4z"/><path d="M5.5 10.2v7.3a1.3 1.3 0 0 0 1.3 1.3h10.4a1.3 1.3 0 0 0 1.3-1.3v-7.3"/><path d="M9.5 13.5h5"/></svg>',
  chev: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 9.5l5 5 5-5"/></svg>',
  playChev: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6.5l6 5.5-6 5.5"/></svg>',
  stop: '<svg viewBox="0 0 24 24"><rect x="6.4" y="6.4" width="11.2" height="11.2" rx="2.6" fill="currentColor"/></svg>',
  arrowup: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19.5V5.5M6.5 11 12 5.5 17.5 11"/></svg>',
  ellipsis: '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="5.6" cy="12" r="1.8"/><circle cx="12" cy="12" r="1.8"/><circle cx="18.4" cy="12" r="1.8"/></svg>',
  terminal: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="5.5" width="16" height="13" rx="1.8"/><path d="M8.2 10l2.6 2.6-2.6 2.6M13.4 15.5h4"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 4.9h-7A1.6 1.6 0 0 0 6.1 6.5v7.6"/><rect x="9" y="8.2" width="9.4" height="10.4" rx="1.6"/><path d="M9 8.2V6.4"/></svg>',
  doctext: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M13.2 3.5H7.3A1.6 1.6 0 0 0 5.7 5.1v13.8a1.6 1.6 0 0 0 1.6 1.6h9.4a1.6 1.6 0 0 0 1.6-1.6V8.2z"/><path d="M13.2 3.5v4.7h4.7"/><path d="M9.2 12.3h5.6M9.2 15.8h4"/></svg>',
  externaldrive: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4.4 13.3h15.2v3.9a1.9 1.9 0 0 1-1.9 1.9H6.3a1.9 1.9 0 0 1-1.9-1.9z"/><path d="M4.4 13.3 6.1 7a1.9 1.9 0 0 1 1.9-1.4h8a1.9 1.9 0 0 1 1.9 1.4l1.7 6.3"/><circle cx="17.1" cy="16.2" r="1" fill="currentColor"/></svg>',
  extlink: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 17 17 7M9.4 7H17v7.6"/></svg>',
  branch: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20V9.2"/><path d="M12 9.2C12 5.9 9.8 4.2 6.8 3.6"/><path d="M12 9.2c0-3.3 2.2-5 5.2-5.6"/><path d="M4.6 2.4 7 3.9 4.6 5.6M19.4 2.4 17 3.9l2.4 1.7"/></svg>',
  sparkle: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.6c.7 4 2.7 6 6.7 6.7-4 .7-6 2.7-6.7 6.7-.7-4-2.7-6-6.7-6.7 4-.7 6-2.7 6.7-6.7z"/><path d="M18.6 15.4c.4 2.2 1.5 3.3 3.7 3.7-2.2.4-3.3 1.5-3.7 3.7-.4-2.2-1.5-3.3-3.7-3.7 2.2-.4 3.3-1.5 3.7-3.7z"/></svg>',
  sliders: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 7.5h7M15.5 7.5H20M4 12h3M11.5 12H20M4 16.5h9M17.5 16.5H20"/><circle cx="13" cy="7.5" r="1.9" fill="#fff"/><circle cx="9" cy="12" r="1.9" fill="#fff"/><circle cx="15" cy="16.5" r="1.9" fill="#fff"/></svg>',
  chart: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 4v16H20"/><path d="M5 18.5 9.5 11l3 3 4.5-6.5 3.5 4"/></svg>',
  info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="12" r="8.4"/><path d="M12 11v5.3" stroke-linecap="round"/><circle cx="12" cy="8" r="1" fill="currentColor" stroke="none"/></svg>',
  person2: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="9" cy="8.6" r="3.1"/><path d="M3.8 19c.7-3.3 2.9-5 5.2-5s4.5 1.7 5.2 5"/><path d="M15.3 11.5c2.7.4 4.4 2.2 4.9 4.8"/></svg>',
  person: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="8.6" r="3.2"/><path d="M6.2 19c.8-3.4 3.1-5.1 5.8-5.1s5 1.7 5.8 5.1"/></svg>',
  check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.3 4.3L18.8 7"/></svg>',
  updown: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 9.5l4-4 4 4M8 14.5l4 4 4-4"/></svg>'
};

function icon(name, size) {
  var body = ICONS[name] || '';
  size = size || 14;
  return body.replace('<svg', '<svg width="' + size + '" height="' + size + '" class="i-' + name + '" aria-hidden="true"');
}

// Inject static icons referenced from index.html.
function injectStaticIcons() {
  var hosts = document.querySelectorAll('[data-icon]');
  for (var i = 0; i < hosts.length; i++) {
    hosts[i].innerHTML = icon(hosts[i].getAttribute('data-icon'), parseInt(hosts[i].getAttribute('data-size'), 10) || 14);
  }
}

// ===== Small helpers ========================================================

function $(sel, root) { return (root || document).querySelector(sel); }
function $all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

function esc(text) {
  return String(text == null ? '' : text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function attr(text) { return esc(text); }

function sleep(ms) {
  return new Promise(function (resolve) { setTimeout(resolve, ms); });
}

var wakePoll = null; // lets the "Retry" button interrupt the backoff sleep
function sleepInterruptible(ms) {
  return new Promise(function (resolve) {
    var done = false;
    wakePoll = function () { if (!done) { done = true; wakePoll = null; resolve(); } };
    setTimeout(function () { if (!done) { done = true; wakePoll = null; resolve(); } }, ms);
  });
}

// Dates: sessions/messages use Apple epoch seconds; dashboard intervals use Unix seconds.
function toJSDate(value) {
  if (value == null || !isFinite(value)) return null;
  var v = Number(value);
  if (v > 1e12) return new Date(v);          // already milliseconds
  if (v > 1e9) return new Date(v * 1000);    // Unix seconds (1970 epoch)
  return new Date((v + EPOCH_OFFSET) * 1000); // Apple epoch seconds (2001)
}

function formatDateTime(value) {
  var d = toJSDate(value);
  if (!d) return '';
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) + ', ' +
    d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
}

function formatTime(value) {
  var d = toJSDate(value);
  if (!d) return '';
  return d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
}

// Mirrors the native app's number formatter (scientific outside 0.001…1e6, 7 significant digits).
function formatMetricNumber(value) {
  if (value == null || !isFinite(value)) return '—';
  var v = Number(value);
  if (v !== 0 && (Math.abs(v) < 0.001 || Math.abs(v) >= 1e6)) {
    var s = v.toExponential(6); // 7 significant digits
    var parts = s.split('e');
    var exp = parseInt(parts[1], 10);
    return parts[0] + 'e' + (exp < 0 ? '-' : '+') + ('0' + Math.abs(exp)).slice(-2);
  }
  return String(Number(v.toPrecision(7)));
}

// ===== API client ===========================================================

function ApiError(status, message) {
  this.status = status;
  this.message = message || ('HTTP ' + status);
  this.name = 'ApiError';
}
ApiError.prototype = Object.create(Error.prototype);

function authHeaders(extra) {
  var h = extra || {};
  if (store.token) h['X-Auth-Token'] = store.token;
  return h;
}

function apiFetch(path, opts) {
  opts = opts || {};
  var headers = authHeaders(opts.headers || {});
  if (opts.body != null && typeof opts.body !== 'string') {
    headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(opts.body);
  } else if (opts.body != null) {
    headers['Content-Type'] = 'application/json';
  }
  return fetch(path, Object.assign({ headers: headers }, opts)).then(function (res) {
    return res.text().then(function (text) {
      var data = null;
      if (text) {
        try { data = JSON.parse(text); } catch (e) { data = text; }
      }
      if (res.status === 401) throw new ApiError(401, 'Sign in required');
      if (!res.ok) {
        var msg = (data && typeof data === 'object' && (data.error || data.message)) || ('HTTP ' + res.status);
        throw new ApiError(res.status, msg);
      }
      return data;
    });
  });
}

function apiGet(path) { return apiFetch(path, { method: 'GET' }); }
function apiPost(path, body) { return apiFetch(path, { method: 'POST', body: body == null ? {} : body }); }

function apiFail(err) {
  if (err && err.status === 401) { promptToken(); return; }
  toast((err && err.message) || 'Request failed', true);
}

// ===== Store ================================================================

var store = {
  token: '',
  version: 0,
  loaded: false,
  projects: [],
  sessions: [],
  selectedProjectID: null,
  selectedSessionID: null,
  provider: localStorage.getItem(PROVIDER_KEY) || 'codex',
  auth: {},
  settings: {},
  server: null,
  search: '',
  followOutput: true,
  keepResearching: {},
  share: readShare(),
  detailsOpen: true, // native `showingProjectDetails` defaults true; newChat() clears it
  pendingNewChat: false,
  pendingSelectSessionID: null,
  refreshing: false,
  dashboard: {}, // projectID -> {data, error, loading, fetchedAt}
  expandedMessages: {}, // messageID -> true (tool / instructions disclosures)
  settingsDraft: null,
  settingsTab: 'Connections',
  cloud: { provider: 'daytona', runTarget: 'mac', keySaved: {}, connected: {}, status: {}, owned: {}, busy: '' },
  lean: { status: 'Stopped', localURL: '', publicURL: '', busy: '' },
  loginProvider: null,
  loginOutput: '',
  loginSpawn: null,
  health: null
};

function readShare() {
  try {
    var raw = JSON.parse(localStorage.getItem(SHARE_KEY) || '{}');
    return raw && typeof raw === 'object' ? raw : {};
  } catch (e) { return {}; }
}

function shareFor(projectID) {
  var s = store.share[projectID];
  if (!s) {
    s = { enabled: false, parallelEnabled: false, agentCount: 3, intervalMinutes: 5 };
    store.share[projectID] = s;
  }
  return s;
}

function saveShare() {
  try { localStorage.setItem(SHARE_KEY, JSON.stringify(store.share)); } catch (e) { /* ignore */ }
}

// ===== Selectors ============================================================

function selectedProject() {
  return store.projects.find(function (p) { return p.id === store.selectedProjectID; }) || null;
}

function selectedSession() {
  return store.sessions.find(function (s) { return s.id === store.selectedSessionID; }) || null;
}

function projectName(project) {
  return (project.name || '').replace(/-/g, ' ');
}

function projectSubtitle(project) {
  if (project.builtinID) return project.category || 'Built-in starter';
  if (project.repository) return project.repository.split('/')[0] || project.repository;
  return 'Local workspace';
}

function projectURL(project) {
  if (project.builtinID) return null; // built-in website is not part of the web state payload
  return project.repository ? 'https://github.com/' + project.repository : null;
}

function openSessionsOf(projectID) {
  return store.sessions
    .filter(function (s) { return s.projectID === projectID && !s.closedAt; })
    .sort(function (a, b) { return (b.createdAt - a.createdAt) || 0; });
}

function closedSessionsOf(projectID) {
  return store.sessions
    .filter(function (s) { return s.projectID === projectID && !!s.closedAt; })
    .sort(function (a, b) { return (b.closedAt - a.closedAt) || 0; });
}

function sessionBusy(s) { return s.state === 'Running' || s.state === 'Preparing'; }

function projectRunning(projectID) {
  return store.sessions.some(function (s) { return s.projectID === projectID && sessionBusy(s); });
}

function runningSessionOf(projectID) {
  return store.sessions.find(function (s) { return s.projectID === projectID && sessionBusy(s); }) || null;
}

function composerBusy() {
  return projectRunning(store.selectedProjectID);
}

function currentProvider() {
  var p = store.provider;
  if (PROVIDERS.indexOf(p) === -1) p = 'codex';
  return p;
}

function authStatus(provider) {
  return store.auth[provider] || 'Checking connection';
}

function isAuthConnected(provider) {
  return authStatus(provider) === 'Connected';
}

// ===== Toast ================================================================

var toastTimer = null;
function toast(message, isError) {
  var el = $('#toast');
  if (!el) return;
  el.textContent = message;
  el.classList.toggle('error', !!isError);
  el.hidden = false;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(function () {
    el.classList.remove('show');
    toastTimer = setTimeout(function () { el.hidden = true; }, 250);
  }, isError ? 4000 : 2600);
}

// ===== Token prompt (inline sheet; no alert()) ==============================

var tokenResolver = null;
var tokenWaiters = [];

function promptToken() {
  if (tokenResolver) { // a prompt is already up: resolve when it closes
    return new Promise(function (resolve) { tokenWaiters.push(resolve); });
  }
  var done = false;
  var host = $('#sheetHost');
  var backdrop = document.createElement('div');
  backdrop.className = 'sheet-backdrop';
  backdrop.innerHTML =
    '<div class="sheet sheet-token" role="dialog" aria-label="Access token">' +
      '<div class="sheet-head"><span class="sheet-title" style="font-size:17px">Access token</span></div>' +
      '<div class="sheet-body" style="padding-top:12px">' +
        '<div class="field"><label for="tokenField">This server requires an access token.</label>' +
        '<input id="tokenField" type="password" placeholder="Paste your token" autocomplete="off"></div>' +
      '</div>' +
      '<div class="sheet-foot"><span class="sheet-note">Stored in this browser.</span><span class="spacer"></span>' +
        '<button class="sheet-secondary" id="tokenCancel">Cancel</button>' +
        '<button class="sheet-primary" id="tokenSave">Save token</button></div>' +
    '</div>';
  host.appendChild(backdrop);
  var field = $('#tokenField', backdrop);
  var inputTimer = setTimeout(function () { if (field) field.focus(); }, 60);
  function close(result) {
    if (done) return;
    done = true;
    clearTimeout(inputTimer);
    backdrop.remove();
    tokenResolver = null;
    if (result) {
      store.token = result;
      try { localStorage.setItem(TOKEN_KEY, result); } catch (e) { /* ignore */ }
    }
    var waiters = tokenWaiters.slice();
    tokenWaiters = [];
    waiters.forEach(function (w) { w(); });
    if (tokenResolve) tokenResolve();
  }
  var tokenResolve = null;
  backdrop.addEventListener('click', function (e) {
    if (e.target === backdrop) close(null);
  });
  $('#tokenSave', backdrop).addEventListener('click', function () {
    close(field.value.trim());
  });
  $('#tokenCancel', backdrop).addEventListener('click', function () { close(null); });
  field.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') { e.preventDefault(); close(field.value.trim()); }
  });
  tokenResolver = close;
  return new Promise(function (resolve) { tokenResolve = resolve; });
}

// ===== State polling ========================================================

var pollStarted = false;

function applyState(data) {
  if (!data || typeof data !== 'object') return;
  if (typeof data.version === 'number') store.version = data.version;
  if (Array.isArray(data.projects)) store.projects = data.projects;
  if (Array.isArray(data.sessions)) store.sessions = data.sessions;
  if (typeof data.selectedProjectID !== 'undefined') store.selectedProjectID = data.selectedProjectID || null;
  if (typeof data.selectedSessionID !== 'undefined') store.selectedSessionID = data.selectedSessionID || null;
  if (data.provider && PROVIDERS.indexOf(data.provider) !== -1) {
    store.provider = data.provider; // server default; the local pick is kept on top below
  }
  store.auth = data.auth || store.auth;
  store.settings = data.settings || store.settings;
  store.server = data.server || store.server;

  if (!localStorage.getItem(PROVIDER_KEY) && data.provider) {
    try { localStorage.setItem(PROVIDER_KEY, currentProvider()); } catch (e) { /* ignore */ }
  }

  // Resolve pending UI intents once the server reflects them.
  if (store.pendingNewChat) {
    var open = openSessionsOf(store.selectedProjectID || '');
    if (open.length) {
      store.selectedSessionID = open[0].id;
      store.pendingNewChat = false;
    }
  }
  if (store.pendingSelectSessionID) {
    var s = store.sessions.find(function (x) { return x.id === store.pendingSelectSessionID && !x.closedAt; });
    if (s) {
      store.selectedSessionID = s.id;
      store.selectedProjectID = s.projectID;
      store.pendingSelectSessionID = null;
    }
  }
  if (store.selectedSessionID) {
    var sel = selectedSession();
    if (!sel || sel.closedAt) store.selectedSessionID = null;
  }
  store.loaded = true;
}

function showConnecting() {
  var body = $('#mainBody');
  if (!body || body.getAttribute('data-mode') === 'connecting') return;
  body.setAttribute('data-mode', 'connecting');
  body.innerHTML =
    '<div class="dash-wrap"><div class="dash-empty" style="margin:28px">' +
      '<div class="spinner"></div>' +
      '<div class="d-title">Connecting…</div>' +
      '<div class="d-detail">Looking for the Auto Research server on <span class="server-url">' + esc(location.host || 'this device') + '</span>.</div>' +
      '<button class="sheet-primary" id="retryConn">Retry now</button>' +
    '</div></div>';
  var btn = $('#retryConn');
  if (btn) btn.addEventListener('click', function () { if (wakePoll) wakePoll(); });
}

function pollLoop() {
  if (pollStarted) return;
  pollStarted = true;
  function step() {
    var since = store.loaded ? store.version : 0;
    apiGet('/api/state' + (since ? '?since=' + encodeURIComponent(since) : '')).then(function (data) {
      applyState(data);
      render();
      step(); // server long-polls up to ~25 s and returns; re-poll immediately
    }).catch(function (err) {
      if (err && err.status === 401) {
        promptToken().then(function () { step(); });
        return;
      }
      if (!store.loaded) showConnecting();
      sleepInterruptible(2000).then(step);
    });
  }
  step();
}

// ===== Actions ==============================================================

function selectProject(id, opts) {
  opts = opts || {};
  var project = store.projects.find(function (p) { return p.id === id; });
  if (!project) return;
  store.selectedProjectID = id;
  if (!opts.keepSession) store.selectedSessionID = null;
  store.detailsOpen = true;
  apiPost('/api/select', { projectID: id }).catch(apiFail);
  render();
}

function selectSession(id) {
  var s = store.sessions.find(function (x) { return x.id === id; });
  if (!s || s.closedAt) return;
  store.selectedSessionID = s.id;
  store.selectedProjectID = s.projectID;
  apiPost('/api/select', { sessionID: id }).catch(apiFail);
  render();
}

function playProject(project, prompt) {
  if (!project) return;
  var body = { projectID: project.id, provider: currentProvider() };
  if (prompt) body.prompt = prompt;
  var share = shareFor(project.id);
  if (share.enabled) {
    body.parallel = true;
    body.agentCount = Number(share.agentCount) || 3;
    body.shareContext = true;
    body.shareEvery = Number(share.intervalMinutes) || 5;
  }
  var keep = Object.keys(store.keepResearching).length > 0;
  if (keep) body.keepResearching = true;
  return apiPost('/api/play', body).then(function (res) {
    store.selectedProjectID = project.id;
    store.detailsOpen = true;
    if (res && res.sessionID) {
      store.selectedSessionID = res.sessionID;
    } else {
      store.pendingNewChat = true;
    }
    render();
    return res;
  }).catch(apiFail);
}

function createChat(projectID, provider) {
  if (!projectID) {
    toast('Select a project first', true);
    return;
  }
  store.pendingNewChat = true;
  store.selectedSessionID = null;
  store.detailsOpen = false; // mirrors native newChat(): welcome screen with the composer ready
  apiPost('/api/chats/create', { projectID: projectID, provider: provider || currentProvider() })
    .then(function (res) {
      if (res && res.sessionID) {
        store.pendingSelectSessionID = res.sessionID;
      }
    }).catch(apiFail);
  render();
}

function closeSession(id) {
  var s = store.sessions.find(function (x) { return x.id === id; });
  apiPost('/api/chats/close', { sessionID: id }).catch(apiFail);
  if (s) s.closedAt = Date.now() / 1000; // optimistic unix-seconds stamp; confirmed by the next poll
  if (store.selectedSessionID === id) store.selectedSessionID = null;
  render();
}

function reopenSession(id) {
  apiPost('/api/chats/reopen', { sessionID: id }).catch(apiFail);
  var s = store.sessions.find(function (x) { return x.id === id; });
  if (s) { s.closedAt = null; store.pendingSelectSessionID = id; }
  render();
}

function refreshCatalog() {
  store.refreshing = true;
  renderSidebar();
  apiPost('/api/projects/refresh', {}).then(function (res) {
    store.refreshing = false;
    if (res && Array.isArray(res.projects)) store.projects = res.projects;
    if (res && res.status && $('#catalogStatus')) {
      store.server = store.server || {};
      store.server.catalogStatus = res.status;
    }
    render();
  }).catch(function (err) {
    store.refreshing = false;
    render();
    apiFail(err);
  });
}

// Adds "owner/name" (or a full GitHub URL) as a repository, otherwise as a local folder path.
function addProjectFromInput(raw, instructions) {
  var input = (raw || '').trim();
  if (!input) return Promise.reject(new ApiError(0, 'Enter a repository or folder path'));
  input = input.replace(/^github\.com\//i, '').replace(/^https?:\/\/(www\.)?github\.com\//i, '').replace(/\.git$/i, '');
  var body;
  if (/^[A-Za-z]:[\\/].*$/.test(input) || input.indexOf('/') === 0 || input.indexOf('~') === 0) {
    body = { localPath: input };
  } else if (/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(input)) {
    body = { repository: input };
  } else {
    body = { localPath: input };
  }
  if (instructions && instructions.trim()) body.instructions = instructions;
  return apiPost('/api/projects/add', body);
}

function stopSession(id) {
  apiPost('/api/chats/stop', { sessionID: id }).catch(apiFail);
}

function chooseProvider(p) {
  if (PROVIDERS.indexOf(p) === -1) return;
  store.provider = p;
  try { localStorage.setItem(PROVIDER_KEY, p); } catch (e) { /* ignore */ }
  var project = selectedProject();
  if (project) createChat(project.id, p); // changing the provider starts a new chat, as in the app
  else render();
}

// ===== Popover menus ========================================================

var openMenu = null;

function closeMenu() {
  if (openMenu) {
    var anchor = openMenu.__anchor;
    if (anchor) anchor.setAttribute('aria-expanded', 'false');
    openMenu.remove();
    openMenu = null;
    document.removeEventListener('mousedown', menuOutside, true);
    document.removeEventListener('keydown', menuKey, true);
  }
}

function menuOutside(e) {
  if (openMenu && !openMenu.contains(e.target) && (!openMenu.__anchor || !openMenu.__anchor.contains(e.target))) {
    closeMenu();
  }
}

function menuKey(e) {
  if (e.key === 'Escape' && openMenu) { e.stopPropagation(); closeMenu(); }
}

function toggleMenu(anchor, alignRight, build) {
  if (openMenu && openMenu.__anchor === anchor) { closeMenu(); return; }
  closeMenu();
  var menu = document.createElement('div');
  menu.className = 'menu-pop';
  menu.setAttribute('role', 'menu');
  menu.__anchor = anchor;
  build(menu);
  document.body.appendChild(menu);
  var r = anchor.getBoundingClientRect();
  var mw = menu.offsetWidth;
  var mh = menu.offsetHeight;
  var left = alignRight ? Math.max(8, Math.min(r.right - mw, window.innerWidth - mw - 8)) : r.left;
  var top = r.bottom + 6;
  if (top + mh > window.innerHeight - 8) top = Math.max(8, r.top - mh - 6);
  menu.style.left = Math.round(left) + 'px';
  menu.style.top = Math.round(top) + 'px';
  anchor.setAttribute('aria-expanded', 'true');
  openMenu = menu;
  setTimeout(function () {
    document.addEventListener('mousedown', menuOutside, true);
    document.addEventListener('keydown', menuKey, true);
  }, 0);
}

function menuItem(menu, label, onClick, opts) {
  opts = opts || {};
  var b = document.createElement('button');
  b.className = 'menu-item' + (opts.active ? ' active' : '');
  b.setAttribute('role', 'menuitem');
  b.innerHTML = (opts.icon ? icon(opts.icon, 13) : '') + '<span>' + esc(label) + '</span>' +
    (opts.active ? '<span class="check">' + icon('check', 13) + '</span>' : '');
  if (opts.disabled) b.disabled = true;
  b.addEventListener('click', function () { closeMenu(); if (onClick) onClick(); });
  menu.appendChild(b);
  return b;
}

function menuSeparator(menu) {
  var d = document.createElement('div');
  d.className = 'menu-sep';
  menu.appendChild(d);
}

// ===== Sidebar ===============================================================

var sideListSig = '';

function renderSidebar() {
  var list = $('#sideList');
  if (!list) return;
  if (openMenu && openMenu.__anchor && list.contains(openMenu.__anchor)) closeMenu();

  var q = store.search.trim().toLowerCase();
  var visible = store.projects.filter(function (p) {
    if (!q) return true;
    var hay = (p.name + ' ' + p.repository + ' ' + p.category + ' ' + (p.builtinID || '')).toLowerCase();
    return hay.indexOf(q) !== -1;
  });
  var builtin = visible.filter(function (p) { return !p.isCustom; });
  var custom = visible.filter(function (p) { return p.isCustom; });

  var projectID = store.selectedProjectID;
  var openChats = projectID ? openSessionsOf(projectID) : [];
  var closedChats = projectID ? closedSessionsOf(projectID) : [];
  var showChats = projectID && (openChats.length > 0 || closedChats.length > 0);

  var sig = store.version + '|' + q + '|' + (projectID || '') + '|' + (store.selectedSessionID || '') +
    '|' + openChats.map(function (s) { return s.id + ':' + s.state + ':' + s.title; }).join(',') +
    '|' + closedChats.length + '|' + visible.map(function (p) { return p.id + ':' + p.name; }).join(',') +
    '|' + (store.refreshing ? 'r' : '');
  if (sig === sideListSig) { updateSidebarFooter(); return; }
  sideListSig = sig;

  var html = [];
  if (!visible.length) {
    html.push('<div class="side-empty">No matching projects</div>');
  }
  html.push(builtin.map(projectRowHTML).join(''));
  if (custom.length) {
    html.push('<div class="side-section-title first" style="padding-top:22px">Your projects</div>');
    html.push(custom.map(projectRowHTML).join(''));
  }
  if (showChats) {
    html.push('<div class="side-section-title first" style="padding-top:22px">Chats in this project</div>');
    html.push(openChats.map(chatRowHTML).join(''));
    if (closedChats.length) {
      html.push(
        '<button class="closed-btn" id="closedMenuBtn" aria-haspopup="menu" aria-expanded="false"' +
        ' aria-label="Closed chats, ' + closedChats.length + ' available">' +
        icon('archive', 13) + '<span>Closed chats (' + closedChats.length + ')</span>' +
        '<span class="chev">' + icon('playChev', 10) + '</span></button>'
      );
    }
  }
  list.innerHTML = html.join('');
  updateSidebarFooter();
}

function projectRowHTML(p) {
  var selected = store.selectedProjectID === p.id;
  var name = esc(projectName(p));
  var running = projectRunning(p.id);
  var provTitle = PROVIDER_TITLES[currentProvider()];
  return (
    '<div class="project-row' + (selected ? ' selected' : '') + '" data-id="' + attr(p.id) + '">' +
      '<button class="row-main" data-select="' + attr(p.id) + '" aria-label="Open project ' + name + '">' +
        '<span class="row-avatar' + (selected ? ' on' : '') + '">' + icon('dotted', 14) + '</span>' +
        '<span class="row-text">' +
          '<span class="row-name">' + name + '</span>' +
          '<span class="row-sub">' + esc(projectSubtitle(p)) + '</span>' +
        '</span>' +
      '</button>' +
      '<button class="row-play' + (running ? ' on' : '') + '" data-play="' + attr(p.id) + '"' +
        ' aria-label="Start research for ' + name + '" title="Start research with ' + esc(provTitle) + '">' +
        icon(running ? 'waveform' : 'play', 10) +
      '</button>' +
    '</div>'
  );
}

function chatRowHTML(s) {
  var selected = store.selectedSessionID === s.id;
  var canClose = !sessionBusy(s);
  return (
    '<div class="chat-row' + (selected ? ' selected' : '') + '">' +
      '<button class="row-main" data-session="' + attr(s.id) + '" aria-label="Open chat ' + esc(s.title) + '">' +
        '<span class="row-icon">' + icon('bubble', 12) + '</span>' +
        '<span class="chat-title">' + esc(s.title) + '</span>' +
        (s.state === 'Running' ? '<span class="running-dot" aria-hidden="true"></span>' : '') +
      '</button>' +
      '<button class="row-close" data-close="' + attr(s.id) + '"' +
        (canClose ? '' : ' disabled') +
        ' aria-label="Close chat: ' + esc(s.title) + '"' +
        ' title="' + esc(canClose ? 'Close chat; keep history' : 'Stop research before closing this chat') + '">' +
        icon('xmark', 10) +
      '</button>' +
    '</div>'
  );
}

function updateSidebarFooter() {
  var server = store.server || {};
  var status = $('#catalogStatus');
  if (status) status.textContent = server.catalogStatus || '';
  var dot = $('#footerDot');
  if (dot) {
    dot.className = 'dot dot-6 ' + (isAuthConnected(currentProvider()) ? 'green' : 'orange');
  }
}

function openClosedChatsMenu(anchor) {
  toggleMenu(anchor, false, function (menu) {
    var closed = closedSessionsOf(store.selectedProjectID || '');
    if (!closed.length) { closeMenu(); return; }
    closed.forEach(function (s) {
      menuItem(menu, s.title + ' · ' + formatDateTime(s.createdAt), function () {
        reopenSession(s.id);
      });
    });
  });
}

// ===== Header ===============================================================

function renderHeader() {
  var header = $('#header');
  if (!header) return;
  var project = selectedProject();
  var session = selectedSession();
  var mode = session ? 'chat' : (project && store.detailsOpen) ? 'dashboard' : 'welcome';
  var prov = session ? session.provider : currentProvider();
  var share = project ? shareFor(project.id) : null;

  var sig = mode + '|' + (project ? project.id + '|' + projectName(project) + '|' + (project.category || '') : '') +
    '|' + (session ? session.id + '|' + session.title + '|' + session.provider + '|' + session.state +
           '|' + (session.sharedAgent ? (session.sharedAgent.sharesContext === false ? 'ind' : 'shared') : '') : '') +
    '|' + prov + '|' + (isAuthConnected(prov) ? 'g' : 'o') + '|' + currentProvider() +
    '|' + (session && sessionBusy(session) ? 'r' : '') + '|' + (projectRunning(store.selectedProjectID) ? 'R' : '') +
    '|' + (share ? JSON.stringify(share) : '') + '|' + (project && project.builtinID ? 'builtin' : '');
  if (header.__sig === sig) return;
  header.__sig = sig;

  var name = session ? esc(session.title) : project ? esc(projectName(project)) : 'Your research workspace';
  var sub = session
    ? (session.sharedAgent
       ? 'Parallel agents · ' + (session.sharedAgent.sharesContext === false ? 'independent' : 'shared chat')
       : 'Single agent · ' + esc(PROVIDER_TITLES[session.provider] || ''))
    : project ? esc(project.category || '') : '';
  var provTitle = PROVIDER_TITLES[prov] || prov;
  var dotClass = isAuthConnected(prov) ? 'green' : 'orange';

  var parts = [];
  parts.push('<div class="h-row">');
  parts.push('<div class="h-title"><div class="h-name">' + name + '</div>');
  if (sub) parts.push('<div class="h-sub">' + sub + '</div>');
  parts.push('</div>');

  if (project && project.builtinID && !session) {
    parts.push('<span class="h-caption" title="This starter uses one local agent. The official judge evaluates kernels remotely.">Local coordinator</span>');
  }
  if (!session && project && !project.builtinID) {
    parts.push('<button class="h-btn" data-action="open-chat">Open chat</button>');
    parts.push('<button class="h-btn primary" data-action="start-research">' + icon('play', 10) + '<span>Start research</span></button>');
  }
  if (session) {
    parts.push('<span class="h-state' + (sessionBusy(session) ? ' running' : '') + '">' + esc(session.state || 'Ready') + '</span>');
    if (sessionBusy(session)) {
      parts.push('<button class="h-icon-btn" data-action="stop" aria-label="Stop research" title="Stop research">' + icon('stop', 13) + '</button>');
    }
  }
  parts.push(
    '<button class="h-menu" data-action="provider" aria-haspopup="menu" aria-expanded="false" aria-label="Provider">' +
      '<span class="dot ' + dotClass + '"></span><span>' + esc(provTitle) + '</span>' +
      '<span class="chev">' + icon('chev', 10) + '</span>' +
    '</button>'
  );
  parts.push('<button class="h-icon-btn" data-action="new-chat" aria-label="New chat" title="New chat">' + icon('edit', 14) + '</button>');
  parts.push('<button class="h-icon-btn" data-action="ellipsis" aria-haspopup="menu" aria-expanded="false" aria-label="More actions">' + icon('ellipsis', 15) + '</button>');
  parts.push('</div>');

  if (!session && project && !project.builtinID) {
    var enabled = !!share.enabled;
    parts.push('<div class="h-share">');
    parts.push(
      '<label class="share-toggle"><span>Share context</span>' +
        '<span class="switch"><input type="checkbox" role="switch" id="shareSwitch"' + (enabled ? ' checked' : '') +
        ' aria-label="Share context between parallel agents"><span class="knob"></span></span></label>'
    );
    if (enabled) {
      parts.push('<span class="share-fields">');
      parts.push('<span class="share-num"><span>Agents</span><input type="number" id="shareAgents" min="2" max="12" step="1" value="' +
        esc(share.agentCount) + '" aria-label="Parallel agent count"></span>');
      parts.push('<span class="share-num min"><span>Share every</span><input type="number" id="shareEvery" min="0.05" step="0.05" value="' +
        esc(share.intervalMinutes) + '" aria-label="Context sharing interval in minutes"><span>min</span></span>');
      parts.push('</span>');
    }
    parts.push('</div>');
  }

  header.innerHTML = parts.join('');

  var sw = $('#shareSwitch', header);
  if (sw) sw.addEventListener('change', function () {
    share.enabled = sw.checked;
    if (sw.checked && !share.parallelEnabled) share.parallelEnabled = true;
    saveShare();
    renderHeader();
  });
  var agents = $('#shareAgents', header);
  if (agents) agents.addEventListener('change', function () {
    share.agentCount = Math.max(2, Math.min(12, parseInt(agents.value, 10) || 3));
    saveShare();
  });
  var every = $('#shareEvery', header);
  if (every) every.addEventListener('change', function () {
    share.intervalMinutes = Math.max(0.05, parseFloat(every.value) || 5);
    saveShare();
  });
}

function openProviderMenu(anchor) {
  toggleMenu(anchor, true, function (menu) {
    var session = selectedSession();
    var cur = session ? session.provider : currentProvider();
    PROVIDERS.forEach(function (p) {
      menuItem(menu, PROVIDER_TITLES[p], function () { chooseProvider(p); }, { active: p === cur });
    });
  });
}

function openEllipsisMenu(anchor) {
  toggleMenu(anchor, true, function (menu) {
    var project = selectedProject();
    var session = selectedSession();
    if (project) {
      menuItem(menu, 'Project details', function () {
        store.selectedSessionID = null;
        store.detailsOpen = true;
        apiPost('/api/select', { projectID: project.id }).catch(apiFail);
        render();
      }, { active: !session && store.detailsOpen });
      menuItem(menu, 'Start research', function () { playProject(project); });
      menuItem(menu, 'Edit instructions', function () { openInstructionsSheet(project.id); });
      if (!project.builtinID) {
        menuItem(menu, 'Edit project details', function () { openProjectEditSheet(project.id); });
      }
      var open = openSessionsOf(project.id);
      if (open.length) menuItem(menu, 'Open latest chat', function () { selectSession(open[0].id); });
      var url = projectURL(project);
      if (url) {
        menuSeparator(menu);
        menuItem(menu, project.builtinID ? 'View challenge' : 'View repository', function () {
          window.open(url, '_blank', 'noopener');
        }, { icon: 'extlink' });
      }
      if (!project.builtinID) {
        menuSeparator(menu);
        menuItem(menu, 'Remove project', function () { openRemoveProjectSheet(project.id); });
      }
      menuSeparator(menu);
    }
    menuItem(menu, 'Show run logs', function () {
      if (!session) return;
      var tokenParam = store.token ? '?token=' + encodeURIComponent(store.token) : '';
      window.open('/api/logs/' + encodeURIComponent(session.id) + tokenParam, '_blank');
    }, { disabled: !session, icon: 'terminal' });
    if (session) {
      menuItem(menu, 'Close chat', function () { closeSession(session.id); }, { disabled: sessionBusy(session) });
    }
    menuSeparator(menu);
    menuItem(menu, 'Connections & settings', function () { openSettingsSheet(); });
  });
}

function headerClick(e) {
  var actionEl = e.target.closest('[data-action]');
  if (!actionEl) return;
  var action = actionEl.getAttribute('data-action');
  if (action === 'provider') { openProviderMenu(actionEl); return; }
  if (action === 'ellipsis') { openEllipsisMenu(actionEl); return; }
  if (action === 'new-chat') { createChat(store.selectedProjectID, currentProvider()); return; }
  if (action === 'stop') {
    var s = selectedSession();
    if (s && sessionBusy(s)) { stopSession(s.id); return; }
    var running = runningSessionOf(store.selectedProjectID || '');
    if (running) stopSession(running.id);
    return;
  }
  if (action === 'open-chat') {
    var project = selectedProject();
    if (!project) return;
    var open = openSessionsOf(project.id);
    if (open.length) selectSession(open[0].id);
    else createChat(project.id, currentProvider());
    return;
  }
  if (action === 'start-research') {
    playProject(selectedProject());
  }
}

// ===== Main content =========================================================

var transcriptSig = '';
var composerSig = '';
var welcomeSig = '';
var followSig = '';
var setupSig = '';
var composerText = '';
var lastMessageCount = {};
var messageTexts = {};

function renderMain() {
  var body = $('#mainBody');
  if (!body || !store.loaded) return;
  var project = selectedProject();
  var session = selectedSession();
  var mode = session ? 'chat' : (project && store.detailsOpen) ? 'dashboard' : 'welcome';
  if (body.__mode !== mode) {
    body.__mode = mode;
    body.setAttribute('data-mode', mode);
    transcriptSig = composerSig = welcomeSig = followSig = setupSig = '';
    if (mode === 'dashboard') {
      body.innerHTML = '<div class="dash-wrap" id="dashWrap"></div>';
    } else if (mode === 'chat') {
      body.innerHTML =
        '<div class="chat-wrap">' +
          '<div class="transcript" id="transcript"><div class="transcript-inner" id="transcriptInner"></div></div>' +
          '<div class="follow-row" id="followRow"></div>' +
          '<div class="composer-zone" id="composerZone"></div>' +
        '</div>';
      $('#transcriptInner').addEventListener('click', transcriptClick);
    } else {
      body.innerHTML =
        '<div class="welcome-wrap">' +
          '<div class="welcome"><div class="welcome-inner" id="welcomeInner"></div></div>' +
          '<div class="composer-zone" id="composerZone"></div>' +
        '</div>';
    }
  }
  if (mode === 'welcome') {
    renderWelcome();
    renderComposer('welcome');
  } else if (mode === 'dashboard') {
    if (project && project.builtinID) renderSetup();
    else renderDashboard();
  } else {
    renderTranscript();
    renderFollowRow();
    renderComposer('chat');
  }
}

// ===== Welcome ==============================================================

function renderWelcome() {
  var host = $('#welcomeInner');
  if (!host) return;
  var project = selectedProject();
  var summary = (project && project.summary) ? project.summary :
    'Give an idea room to grow. Your agent can explore the code, run experiments, and keep the evidence together.';
  var url = project ? projectURL(project) : null;
  var exploreLabel = project ? (project.builtinID ? 'Explore challenge' : 'Explore repository') : null;
  var sig = (project ? project.id + '|' + (project.summary || '') + '|' + (url || '') : 'none');
  if (sig === welcomeSig) return;
  welcomeSig = sig;

  host.innerHTML =
    '<div class="welcome-tagline">' + icon('sparkle', 14) + '<span>A little curiosity. A lot of possibility.</span></div>' +
    '<h1 class="welcome-headline">Where will your\nresearch lead?</h1>' +
    '<p class="welcome-summary">' + esc(summary) + '</p>' +
    '<div class="welcome-actions">' +
      '<button class="capsule" id="welcomePlay"' + (project ? '' : ' disabled') + '>' +
        icon('play', 12) + '<span>Start research</span></button>' +
      (url ? '<a class="capsule secondary" href="' + attr(url) + '" target="_blank" rel="noreferrer">' +
        icon('extlink', 12) + '<span>' + esc(exploreLabel) + '</span></a>' : '') +
    '</div>' +
    '<div class="welcome-hints">' +
      hintHTML('doctext', 'Repository-led', 'Starts from its research guide') +
      hintHTML('terminal', 'Your CLI, your account', 'Codex, Claude or OpenCode underneath') +
      hintHTML('externaldrive', 'A durable workspace', 'Chats, code, and experiment logs') +
    '</div>';

  var play = $('#welcomePlay', host);
  if (play) play.addEventListener('click', function () { playProject(selectedProject()); });
}

function hintHTML(sym, title, detail) {
  return '<div class="hint"><span class="h-icon">' + icon(sym, 17) + '</span>' +
    '<span class="h-title">' + esc(title) + '</span>' +
    '<span class="h-detail">' + esc(detail) + '</span></div>';
}

// ===== Transcript ===========================================================

function assistantVisibleText(text) {
  return String(text || '').split('\n')
    .filter(function (line) { return line.indexOf('RESEARCH_STATUS: ') !== 0; })
    .join('\n');
}

function renderTranscript() {
  var host = $('#transcriptInner');
  var session = selectedSession();
  if (!host || !session) return;
  var msgs = session.messages || [];
  var last = msgs.length ? msgs[msgs.length - 1] : null;
  var busy = sessionBusy(session);
  var sig = session.id + '|' + msgs.length + '|' + (last ? last.id : '') + '|' + session.state + '|' + (busy ? 'b' : '');
  if (sig === transcriptSig) return;
  transcriptSig = sig;

  var countChanged = lastMessageCount[session.id] !== msgs.length;
  lastMessageCount[session.id] = msgs.length;

  var html = [];
  for (var i = 0; i < msgs.length; i++) html.push(messageHTML(msgs[i], session));
  if (busy) {
    html.push('<div class="running-row"><div class="spinner"></div><span>' +
      (session.state === 'Preparing' ? 'Preparing workspace…' : 'Researching…') + '</span></div>');
  }
  host.innerHTML = html.join('');
  messageTexts = {};
  for (var j = 0; j < msgs.length; j++) messageTexts[msgs[j].id] = msgs[j].text;

  if (store.followOutput || countChanged) {
    var transcript = $('#transcript');
    if (transcript) transcript.scrollTop = transcript.scrollHeight;
  }
}

function messageHTML(m, session) {
  var role = m.role || 'assistant';
  var speaker = m.speaker || null;
  var mid = attr(m.id);
  var out = '<div class="msg" data-msg="' + mid + '">';
  var providerTitle = PROVIDER_TITLES[session.provider] || 'Assistant';

  if (role === 'tool') {
    var lines = String(m.text || '').split('\n');
    var label = lines[0] || 'Tool activity';
    var expanded = !!store.expandedMessages[m.id];
    out += speaker ? '<div class="msg-head">' + esc(speaker) + '</div>' : '';
    out +=
      '<div class="tool-card' + (expanded ? ' expanded' : '') + '">' +
        '<div class="tool-head">' +
          '<button class="tool-toggle" data-toggle="' + mid + '" aria-expanded="' + (expanded ? 'true' : 'false') + '">' +
            '<span class="t-icon">' + icon('terminal', 13) + '</span>' +
            '<span class="t-label">' + esc(label) + '</span>' +
            '<span class="spacer"></span>' +
            '<span class="chev">' + icon('playChev', 10) + '</span>' +
          '</button>' +
          '<button class="tool-copy" data-copy="' + mid + '" aria-label="Copy message" title="Copy message">' +
            icon('copy', 11) + '</button>' +
        '</div>' +
        '<div class="tool-body"' + (expanded ? '' : ' hidden') + '>' + esc(m.text || '') + '</div>' +
      '</div>';
    return out + '</div>';
  }

  if (role === 'status') {
    out += speaker ? '<div class="msg-head">' + esc(speaker) + '</div>' : '';
    out += '<div class="status-line">' + icon('branch', 13) + '<span>' + esc(m.text || '') + '</span></div>';
    return out + '</div>';
  }

  // user / assistant / error
  var heading = role === 'user' ? (speaker ? 'You → ' + speaker : 'You')
    : role === 'error' ? (speaker ? speaker + ' · Needs attention' : 'Needs attention')
    : (speaker || providerTitle);
  var text = role === 'assistant' ? assistantVisibleText(m.text) : String(m.text || '');
  var long = role === 'user' && text.length > 1000;
  var expanded = !!store.expandedMessages[m.id];
  var bubble = role === 'user' ? 'msg-bubble user' : role === 'error' ? 'msg-bubble error' : '';

  out += bubble ? '<div class="' + bubble + '"><div class="msg-pad">' : '';
  out += '<div class="msg-head' + (role === 'error' ? ' error' : '') + '">' +
    '<span>' + esc(heading) + '</span>' +
    '<button class="msg-copy" data-copy="' + mid + '" aria-label="Copy message" title="Copy message">' +
      icon('copy', 11) + '</button>' +
  '</div>';
  if (long) {
    out += '<button class="instructions-toggle' + (expanded ? ' expanded' : '') + '" data-toggle="' + mid + '"' +
      ' aria-expanded="' + (expanded ? 'true' : 'false') + '">' +
      '<span class="chev">' + icon('playChev', 10) + '</span><span>Research instructions</span></button>';
    out += '<div class="instructions-body"' + (expanded ? '' : ' hidden') + '>' + esc(text) + '</div>';
  } else {
    out += '<div class="msg-body">' + esc(text) + '</div>';
  }
  out += bubble ? '</div></div>' : '';
  return out + '</div>';
}

function transcriptClick(e) {
  var copyBtn = e.target.closest('[data-copy]');
  if (copyBtn) {
    var text = messageTexts[copyBtn.getAttribute('data-copy')] || '';
    copyText(text);
    return;
  }
  var toggleBtn = e.target.closest('[data-toggle]');
  if (toggleBtn) {
    var id = toggleBtn.getAttribute('data-toggle');
    var expanded = !store.expandedMessages[id];
    store.expandedMessages[id] = expanded;
    var card = toggleBtn.closest('.tool-card');
    var body = toggleBtn.closest('.msg').querySelector('.tool-body, .instructions-body');
    toggleBtn.classList.toggle('expanded', expanded);
    toggleBtn.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    if (body) body.hidden = !expanded;
    if (card) card.classList.toggle('expanded', expanded);
    if (expanded && body) body.hidden = false;
  }
}

function copyText(text) {
  var finish = function () { toast('Copied to clipboard'); };
  var fail = function () { toast('Copy failed', true); };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(finish, fail);
  } else {
    try {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
      finish();
    } catch (e) { fail(); }
  }
}

// ===== Follow row ===========================================================

function renderFollowRow() {
  var host = $('#followRow');
  var session = selectedSession();
  if (!host || !session) return;
  var sig = session.id;
  if (sig === followSig) return;
  followSig = sig;
  host.innerHTML =
    '<label><input type="checkbox" id="followOutputChk"' + (store.followOutput ? ' checked' : '') + '>' +
      '<span>Follow output</span></label>' +
    '<span class="spacer"></span>' +
    '<label title="Send the editable continuation prompt after each successful turn until you stop. Uses your CLI account and configured resources.">' +
      '<input type="checkbox" id="keepResearchingChk"' + (store.keepResearching[session.id] ? ' checked' : '') + '>' +
      '<span>Keep researching</span></label>';
  $('#followOutputChk', host).addEventListener('change', function (e) {
    store.followOutput = e.target.checked;
    if (store.followOutput) {
      var t = $('#transcript');
      if (t) t.scrollTop = t.scrollHeight;
    }
  });
  $('#keepResearchingChk', host).addEventListener('change', function (e) {
    store.keepResearching[session.id] = e.target.checked;
  });
}

// ===== Composer =============================================================

function renderComposer(mode) {
  var host = $('#composerZone');
  if (!host) return;
  var project = selectedProject();
  var session = selectedSession();
  var busy = composerBusy();
  var repoLabel = project ? (project.repository || projectName(project) || 'Select a project') : 'Select a project';
  var sig = (session ? session.id : 'none|' + (project ? project.id : '')) + '|' + mode + '|' + (busy ? 'busy' : '') + '|' + repoLabel;
  if (sig === composerSig) return;
  composerSig = sig;

  var placeholder = busy ? 'Draft a follow-up while research runs…' : 'Ask a question, or give your research a direction…';
  var prov = session ? session.provider : currentProvider();
  var meta = PROVIDER_TITLES[prov] + ' · ' + authStatus(prov) + ' · Uses your CLI’s configured model';

  host.innerHTML =
    '<div class="composer-stack">' +
      '<div class="composer">' +
        '<div class="composer-input-wrap">' +
          '<div class="composer-placeholder" id="composerPlaceholder">' + esc(placeholder) + '</div>' +
          '<textarea class="composer-input" id="composerInput" rows="2" aria-label="Message"' +
            ' placeholder="" spellcheck="false"></textarea>' +
        '</div>' +
        '<div class="composer-bottom">' +
          '<div class="composer-repo"><span class="r-icon">' + icon('folder', 12) + '</span>' +
            '<span class="r-label">' + esc(repoLabel) + '</span></div>' +
          '<span class="spacer"></span>' +
          (busy
            ? '<button class="stop-btn" id="composerStop" aria-label="Stop research" title="Stop this run">' + icon('stop', 13) + '</button>'
            : '<button class="send-btn" id="composerSend" aria-label="Send message"' +
                ' title="Enter to send · Shift+Enter for a new line">' + icon('arrowup', 14) + '</button>') +
        '</div>' +
      '</div>' +
      (busy ? '<div class="composer-caption">Stop research to send your draft.</div>' : '') +
      '<div class="composer-meta">' + esc(meta) + '</div>' +
    '</div>';

  var input = $('#composerInput', host);
  input.value = composerText;
  input.addEventListener('input', function () {
    composerText = input.value;
    autoResizeComposer(input);
    updateSendState();
  });
  input.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      sendComposer();
    }
  });
  input.addEventListener('focus', updateSendState);
  autoResizeComposer(input);

  var send = $('#composerSend', host);
  if (send) send.addEventListener('click', sendComposer);
  var stop = $('#composerStop', host);
  if (stop) stop.addEventListener('click', function () {
    var running = runningSessionOf(store.selectedProjectID || '');
    if (running) stopSession(running.id);
    var s = selectedSession();
    if (s && sessionBusy(s)) stopSession(s.id);
  });
  updateSendState();
}

function autoResizeComposer(input) {
  input.style.height = 'auto';
  var h = Math.max(56, Math.min(94, input.scrollHeight));
  input.style.height = h + 'px';
  input.style.overflowY = input.scrollHeight > 94 ? 'auto' : 'hidden';
}

function updateSendState() {
  var send = $('#composerSend');
  var placeholder = $('#composerPlaceholder');
  var busy = composerBusy();
  if (placeholder) placeholder.hidden = composerText.length > 0;
  if (send) {
    var ready = !busy && composerText.trim().length > 0 && selectedProject();
    send.classList.toggle('ready', ready);
    send.disabled = !ready;
  }
}

function syncComposer() {
  var input = $('#composerInput');
  if (input) {
    input.value = composerText;
    autoResizeComposer(input);
    input.focus();
  }
  updateSendState();
}

function sendComposer() {
  var text = composerText.trim();
  if (!text || composerBusy()) return;
  var project = selectedProject();
  var session = selectedSession();
  if (session) {
    composerText = '';
    syncComposer();
    apiPost('/api/chats/send', { sessionID: session.id, text: text,
                                 keepResearching: !!store.keepResearching[session.id] }).catch(apiFail);
  } else if (project) {
    composerText = '';
    syncComposer();
    // mirrors the native send(): with no open chat the request starts a new
    // chat titled with the request itself
    apiPost('/api/chats/send', { sessionID: '', projectID: project.id, text: text })
      .then(function (res) {
        if (res && res.sessionID) {
          store.pendingSelectSessionID = res.sessionID;
          store.detailsOpen = false;
        }
      }).catch(apiFail);
  }
}

// ===== Built-in setup page ==================================================

var setupCache = {};

function setupEntry(projectID) {
  var e = setupCache[projectID];
  if (!e) {
    e = { loading: false, loaded: false, data: null, error: null, saving: false };
    setupCache[projectID] = e;
  }
  return e;
}

function loadSetup(projectID, force) {
  var entry = setupEntry(projectID);
  if (entry.loading) return;
  if (!force && entry.loaded) return;
  entry.loading = true;
  apiGet('/api/setup?projectID=' + encodeURIComponent(projectID)).then(function (res) {
    entry.loading = false;
    entry.loaded = true;
    entry.data = res || {};
    entry.error = null;
    renderSetup();
  }).catch(function (err) {
    entry.loading = false;
    entry.loaded = true;
    entry.error = (err && err.message) || 'The setup form is unavailable.';
    renderSetup();
  });
}

function setupDraftLocal(projectID) {
  var entry = setupEntry(projectID);
  if (!entry.data) return null;
  if (!entry.data.draft) entry.data.draft = {};
  return entry.data.draft;
}

function renderSetup() {
  var wrap = $('#dashWrap');
  var project = selectedProject();
  if (!wrap || !project) return;
  var entry = setupEntry(project.id);
  loadSetup(project.id);

  var sig = project.id + '|' + projectName(project) + '|' + (project.summary || '') +
    '|' + (entry.loading ? 'L' : '') + '|' + (entry.error || '') +
    '|' + (entry.data ? JSON.stringify(entry.data.draft || {}).length : '');
  if (sig === setupSig) return;
  setupSig = sig;

  var parts = ['<div class="dash-inner">'];
  parts.push(
    '<div class="dash-head">' +
      '<div class="dash-title-block"><h2 class="dash-title">' + esc(projectName(project)) + '</h2>' +
        (project.summary ? '<p class="dash-summary">' + esc(project.summary) + '</p>' : '') + '</div>' +
      '<div class="dash-head-actions">' +
        (project.website ? '<a class="h-icon-btn" href="' + attr(project.website) + '" target="_blank" rel="noreferrer" aria-label="Open the challenge page" title="Open the challenge page">' + icon('extlink', 15) + '</a>' : '') +
        '<button class="h-icon-btn" id="setupGuideBtn" aria-label="Edit the workspace guide" title="Edit the workspace guide">' + icon('edit', 15) + '</button>' +
      '</div>' +
    '</div>');

  if (entry.error) {
    parts.push('<div class="dash-empty"><div class="d-title">' + esc(entry.error) + '</div>' +
      '<button class="sheet-primary" id="setupRetry">Retry</button></div>');
  } else if (entry.data) {
    parts.push(setupFormHTML(project, entry));
  } else {
    parts.push('<div class="dash-empty"><div class="spinner"></div><div class="d-detail">Loading setup…</div></div>');
  }
  parts.push('</div>');
  wrap.innerHTML = parts.join('');
  wireSetup(project, entry);
}

function setupFormHTML(project, entry) {
  var draft = entry.data.draft || {};
  var prompts = entry.data.suggestedPrompts || [];
  var scored = !!draft.allowScoredSubmissions;
  var preview = entry.data.prompt ? '<pre class="prompt-preview">' + esc(entry.data.prompt) + '</pre>'
    : (entry.data.promptError ? '<p class="setup-warning">' + esc(entry.data.promptError) + '</p>' : '');
  var picker = prompts.map(function (p, i) {
    return '<option value="' + i + '"' + (p === draft.researchPrompt ? ' selected' : '') + '>' +
      esc(p.length > 90 ? p.slice(0, 90) + '…' : p) + '</option>';
  }).join('');
  return '<div class="setup-form">' +
    '<p class="setup-lede"><span class="setup-lede-icon">' + icon('laptop', 15) + '</span>' +
      'No GPU rental required. Your local agent edits the public starter code. Open Kernel Challenge runs checks and scores on its GPUs using your account’s quota. A verified leaderboard improvement is the goal; results are not guaranteed.</p>' +
    '<p class="setup-lede">Start with your own OKC account and accept the rules in the terminal. The agent never enters credentials.</p>' +
    '<div class="setup-account">' +
      '<div class="field"><label for="setupAccount">Your GitHub/OKC username</label>' +
        '<input id="setupAccount" data-setup="expectedAccount" placeholder="The username the run verifies" autocomplete="off"></div>' +
      '<div class="account-actions">' +
        '<button class="sheet-secondary" id="setupVerify">Refresh account</button>' +
        '<button class="sheet-secondary" id="setupChoose">Choose account</button>' +
      '</div>' +
      '<div class="account-status" id="setupAccountStatus"></div>' +
    '</div>' +
    '<div class="field"><label for="setupSource">Kernel source (optional)</label>' +
      '<input id="setupSource" data-setup="sourcePath" placeholder="Leave blank to start from the public starter"></div>' +
    '<div class="field-row">' +
      '<div class="field"><label for="setupChecks">Official checks per day</label>' +
        '<input id="setupChecks" data-setup="checkBudget" inputmode="numeric" placeholder="0 means local analysis only"></div>' +
      '<div class="field narrow">' +
        '<label><input type="checkbox" data-setup="useSuperExperiments"' + (draft.useSuperExperiments !== false ? ' checked' : '') + '>' +
          '<span>Use Super Experiments pre-flight runs</span></label>' +
      '</div>' +
    '</div>' +
    '<div class="field"><label><input type="checkbox" data-setup="allowScoredSubmissions"' + (scored ? ' checked' : '') + '>' +
      '<span>Allow scored submissions</span></label></div>' +
    (scored ?
      '<div class="field-row">' +
        '<div class="field"><label for="setupScore">Scored submissions</label>' +
          '<input id="setupScore" data-setup="scoreBudget" inputmode="numeric" placeholder="Positive whole number"></div>' +
        '<div class="field"><label for="setupCriterion">When a scored submission is allowed</label>' +
          '<input id="setupCriterion" data-setup="scoreCriterion" placeholder="Your first-baseline or improvement criterion"></div>' +
      '</div>' : '') +
    '<div class="field"><label for="setupPrompt">Research prompt</label>' +
      (picker ? '<select id="setupPromptPicker" class="metric-select" aria-label="Suggested prompts"><option value="-1">Suggested prompts</option>' + picker + '</select>' : '') +
      '<textarea id="setupPrompt" data-setup="researchPrompt" rows="4" placeholder="What the run should do"></textarea></div>' +
    '<details class="setup-preview"' + (entry.data.prompt ? '' : ' open') + '>' +
      '<summary>Review request</summary>' + preview + '</details>' +
    '<div class="setup-actions">' +
      '<button class="sheet-secondary" id="setupSave">Save setup</button>' +
      '<button class="sheet-secondary" id="setupUseChat" title="Copy the reviewed request to the composer">Use in chat</button>' +
      '<button class="sheet-primary" id="setupPlay">' + icon('play', 11) + '<span>Start research</span></button>' +
    '</div>' +
    (entry.data.guidePath ?
      '<details class="setup-guide"><summary>Installation and login instructions</summary>' +
      '<pre class="prompt-preview">' + esc(entry.data.loginInstructions || '') + '</pre></details>' : '') +
  '</div>';
}

function wireSetup(project, entry) {
  var wrap = $('#dashWrap');
  if (!wrap) return;
  var retry = $('#setupRetry', wrap);
  if (retry) retry.addEventListener('click', function () { setupCache[project.id].loaded = false; loadSetup(project.id, true); });
  var guide = $('#setupGuideBtn', wrap);
  if (guide) guide.addEventListener('click', function () { openInstructionsSheet(project.id); });
  $all('[data-setup]', wrap).forEach(function (input) {
    var key = input.getAttribute('data-setup');
    var draft = setupDraftLocal(project.id);
    if (draft && draft[key] != null) {
      if (input.type === 'checkbox') input.checked = !!draft[key];
      else input.value = draft[key];
    }
    var evt = input.type === 'checkbox' ? 'change' : 'input';
    input.addEventListener(evt, function () {
      var d = setupDraftLocal(project.id) || {};
      d[key] = input.type === 'checkbox' ? input.checked : input.value;
      if (key === 'allowScoredSubmissions') { setupSig = ''; renderSetup(); }
    });
  });
  var picker = $('#setupPromptPicker', wrap);
  if (picker) picker.addEventListener('change', function () {
    if (picker.value === '-1') return;
    var prompt = (entry.data.suggestedPrompts || [])[parseInt(picker.value, 10)] || '';
    var field = $('#setupPrompt', wrap);
    if (field) { field.value = prompt; var d = setupDraftLocal(project.id); if (d) d.researchPrompt = prompt; }
  });
  var verify = $('#setupVerify', wrap);
  if (verify) verify.addEventListener('click', function () {
    verify.textContent = 'Checking…';
    apiPost('/api/setup/verify-account', {}).then(function (res) {
      var status = $('#setupAccountStatus', wrap);
      var draft = setupDraftLocal(project.id) || {};
      var account = (res && res.account) || '';
      var expected = (draft.expectedAccount || '').replace(/^@/, '').toLowerCase();
      if (status) {
        status.textContent = 'Signed in as ' + account;
        status.classList.toggle('mismatch', expected && account !== expected);
        if (expected && account !== expected) {
          status.textContent = 'The signed-in account (' + account +
            ') differs from your expected username. Choose your account, then refresh.';
        }
      }
      verify.textContent = 'Refresh account';
      toast('OKC account verified');
    }).catch(function (err) {
      verify.textContent = 'Refresh account';
      var status = $('#setupAccountStatus', wrap);
      if (status) status.textContent = (err && err.message) || 'Not verified';
      apiFail(err);
    });
  });
  var choose = $('#setupChoose', wrap);
  if (choose) choose.addEventListener('click', function () {
    apiPost('/api/setup/login-script', { projectID: project.id, chooseAccount: true }).then(function (res) {
      toast(res && res.spawned ? 'A terminal window opened on the host device' : 'Open the script path shown below');
      openLoginNote(res);
    }).catch(apiFail);
  });
  var save = $('#setupSave', wrap);
  if (save) save.addEventListener('click', function () { saveSetupDraft(project.id, entry); });
  var useChat = $('#setupUseChat', wrap);
  if (useChat) useChat.addEventListener('click', function () {
    saveSetupDraft(project.id, entry, true).then(function () {
      composerText = entry.data.prompt || '';
      store.selectedSessionID = null;
      store.detailsOpen = false;
      render();
    }).catch(function (err) { apiFail(err); });
  });
  var play = $('#setupPlay', wrap);
  if (play) play.addEventListener('click', function () {
    saveSetupDraft(project.id, entry, true).then(function () {
      playProject(project);
    }).catch(function (err) { apiFail(err); });
  });
}

function saveSetupDraft(projectID, entry, quiet) {
  var draft = setupDraftLocal(projectID) || {};
  entry.saving = true;
  return apiPost('/api/setup', { projectID: projectID, draft: draft }).then(function (res) {
    entry.saving = false;
    if (res && res.draft) entry.data.draft = res.draft;
    setupSig = '';
    if (!quiet) toast('Setup saved');
    renderSetup();
    return res;
  }).catch(function (err) {
    entry.saving = false;
    setupSig = '';
    renderSetup();
    throw err;
  });
}

// ===== Dashboard ============================================================

var dashSig = '';

function getDashEntry(projectID) {
  var e = store.dashboard[projectID];
  if (!e) {
    e = { loading: false, data: null, error: null, notFound: false, fetchedAt: 0, groupID: null, observed: false };
    store.dashboard[projectID] = e;
  }
  return e;
}

function loadDashboard(projectID, force) {
  var entry = getDashEntry(projectID);
  if (entry.loading) return;
  var stale = !entry.fetchedAt || (Date.now() - entry.fetchedAt > 60000);
  var cooled = entry.fetchedAt && (Date.now() - entry.fetchedAt < 20000);
  if (!force && entry.data && !stale) return;
  if (!force && !entry.data && cooled) return;
  entry.loading = true;
  renderDashboard();
  apiGet('/api/dashboard?projectID=' + encodeURIComponent(projectID)).then(function (data) {
    entry.data = data;
    entry.error = null;
    entry.notFound = false;
    entry.fetchedAt = Date.now();
    entry.loading = false;
    if (data && data.groups && data.groups.length) {
      var stillThere = data.groups.some(function (g) { return g.id === entry.groupID; });
      if (!stillThere) entry.groupID = null;
    }
    renderDashboard();
  }).catch(function (err) {
    entry.loading = false;
    entry.fetchedAt = Date.now();
    if (err && err.status === 401) { promptToken(); return; }
    if (err && err.status === 404) { entry.notFound = true; entry.error = null; }
    else { entry.error = (err && err.message) || 'Update unavailable'; }
    renderDashboard();
  });
}

function renderDashboard() {
  var wrap = $('#dashWrap');
  var project = selectedProject();
  if (!wrap || !project) return;
  var entry = getDashEntry(project.id);
  loadDashboard(project.id);

  var data = entry.data;
  var groups = (data && Array.isArray(data.groups)) ? data.groups : [];
  var group = groups.find(function (g) { return g.id === entry.groupID; }) || groups[0] || null;
  var mode = entry.observed ? 'observed' : 'estimated';
  var url = projectURL(project);

  var sig = project.id + '|' + projectName(project) + '|' + (project.summary || '') + '|' +
    (entry.loading ? 'L' : '') + '|' + (entry.error || '') + '|' + (entry.notFound ? '404' : '') + '|' +
    (data ? (data.as_of || '') + '|' + (data.claimStatus || '') + '|' + groups.length + '|' + pullsCount(data) : 'nodata') +
    '|' + (group ? group.id + ':' + mode : '') + '|' + entry.observed + '|' + wrap.clientWidth;
  if (sig === dashSig) return;
  dashSig = sig;

  var parts = ['<div class="dash-inner">'];

  parts.push(
    '<div class="dash-head">' +
      '<div class="dash-title-block"><h2 class="dash-title">' + esc(projectName(project)) + '</h2>' +
        (project.summary ? '<p class="dash-summary">' + esc(project.summary) + '</p>' : '') + '</div>' +
      '<div class="dash-head-actions">' +
        (url ? '<a class="h-icon-btn" href="' + attr(url) + '" target="_blank" rel="noreferrer" aria-label="Open repository" title="Open repository">' + icon('extlink', 15) + '</a>' : '') +
        '<button class="h-icon-btn" id="dashRefresh" aria-label="Refresh chart data" title="Refresh chart data"' + (entry.loading ? ' disabled' : '') + '>' + icon('refresh', 15) + '</button>' +
      '</div>' +
    '</div>'
  );

  if (data && group) {
    parts.push(chartCardHTML(project, entry, group, mode));
    parts.push(recentActivityHTML(data));
  } else if (entry.loading) {
    parts.push('<div class="dash-empty"><div class="spinner"></div><div class="d-detail">Loading chart…</div></div>');
  } else {
    var title = entry.error ? 'Chart data is unavailable' : 'No chart published yet';
    var detail = entry.error ? 'Refresh to try again.' : 'You can explore the repository before starting.';
    parts.push('<div class="dash-empty"><span class="d-icon">' + icon('chart', 28) + '</span>' +
      '<div class="d-title">' + esc(title) + '</div><div class="d-detail">' + esc(detail) + '</div>' +
      (entry.error ? '<button class="sheet-primary" id="dashRetry">Retry</button>' : '') + '</div>');
  }

  parts.push(dashFootHTML(data, entry));
  parts.push('</div>');
  wrap.innerHTML = parts.join('');

  var refresh = $('#dashRefresh', wrap);
  if (refresh) refresh.addEventListener('click', function () { loadDashboard(project.id, true); });
  var retry = $('#dashRetry', wrap);
  if (retry) retry.addEventListener('click', function () { loadDashboard(project.id, true); });

  if (data && group) {
    wireChartCard(project, entry, group, mode);
    renderChart();
  }
}

function pullsCount(data) {
  return (data && Array.isArray(data.pulls) ? data.pulls.length : 0);
}

function chartCardHTML(project, entry, group, mode) {
  var hist = mode === 'observed' ? group.observed : group.estimated;
  var intervals = normIntervals(hist ? hist.points : []);
  var value = hist ? hist.value : null;
  var sources = (group.latest && Array.isArray(group.latest.sources)) ? group.latest.sources : [];
  var src = sources.length ? sources[0] : null;
  var groups = entry.data.groups;

  var links;
  if (src) {
    links = '<a href="' + attr(src) + '" target="_blank" rel="noreferrer">Open proposal</a>' +
      '<a href="' + attr(src) + '" target="_blank" rel="noreferrer">Source&nbsp;<span class="ext">' + icon('extlink', 9) + '</span></a>';
  } else {
    links = '<span>Source claim</span>';
  }

  var picker;
  if (groups.length > 1) {
    picker = '<div class="metric-select-wrap"><select class="metric-select" id="metricSelect" aria-label="Metric">' +
      groups.map(function (g) {
        return '<option value="' + attr(g.id) + '"' + (g.id === group.id ? ' selected' : '') + '>' + esc(g.name) + '</option>';
      }).join('') + '</select></div>';
  } else {
    picker = '<div class="metric-solo">' + esc(group.name) + '</div>';
  }

  var directionCaption = group.direction === 'up' ? 'Higher is better'
    : group.direction === 'down' ? 'Lower is better' : 'Backend metric';
  var rightCaption = entry.observed ? 'Observed claims' : 'Claims · estimated timing';

  return '<div class="chart-card">' +
    '<div class="chart-top">' +
      '<div style="min-width:0">' +
        '<div class="metric-number">' + esc(formatMetricNumber(value)) + '</div>' +
        '<div class="metric-links">' + links + '</div>' +
      '</div>' +
      '<div class="segmented" role="group" aria-label="Backend claim history with estimated dates, or changes observed by the backend">' +
        '<button id="segEstimated" aria-pressed="' + (!entry.observed) + '">Estimated</button>' +
        '<button id="segObserved" aria-pressed="' + (!!entry.observed) + '">Observed</button>' +
      '</div>' +
    '</div>' +
    '<div class="metric-picker-row">' + picker + '</div>' +
    (intervals.length
      ? '<div class="chart-host" id="chartHost"></div>'
      : '<div class="dash-empty" style="min-height:230px;background:none"><div class="d-detail">No history available for this view.</div></div>') +
    '<div class="chart-captions"><span>' + esc(directionCaption) + '</span><span class="spacer"></span><span>' +
      esc(rightCaption) + '</span></div>' +
  '</div>';
}

function wireChartCard(project, entry, group, mode) {
  var wrap = $('#dashWrap');
  var segE = $('#segEstimated', wrap);
  var segO = $('#segObserved', wrap);
  if (segE) segE.addEventListener('click', function () {
    entry.observed = false; renderDashboard();
  });
  if (segO) segO.addEventListener('click', function () {
    entry.observed = true; renderDashboard();
  });
  var select = $('#metricSelect', wrap);
  if (select) select.addEventListener('change', function () {
    entry.groupID = select.value;
    renderDashboard();
  });
}

function recentActivityHTML(data) {
  var pulls = (Array.isArray(data.pulls) ? data.pulls : []).filter(function (p) { return p && p.html_url; });
  if (!pulls.length) return '';
  var rows = pulls.slice(0, 3).map(function (p) {
    var status = p.status || (p.merged_at ? 'Merged' : (p.state === 'closed' ? 'Closed' : 'Open'));
    return '<a class="pull-row" href="' + attr(p.html_url) + '" target="_blank" rel="noreferrer">' +
      '<span class="pull-num">#' + esc(p.number) + '</span>' +
      '<span class="pull-title">' + esc(p.title || '') + '</span>' +
      '<span class="spacer"></span>' +
      '<span class="pull-status">' + esc(status) + '</span>' +
      '<span class="pull-ext">' + icon('extlink', 9) + '</span>' +
    '</a>';
  }).join('');
  return '<div class="activity">' +
    '<div class="activity-head"><span class="activity-title">Recent activity</span>' +
      '<span class="activity-count">' + pulls.length + ' PRs</span></div>' + rows + '</div>';
}

function dashFootHTML(data, entry) {
  var project = selectedProject();
  var claimStatus = (data && data.claimStatus) ? String(data.claimStatus) : '';
  var stale = claimStatus && claimStatus !== 'ok' && claimStatus !== 'fresh';
  var asOf = data && data.as_of ? parseISO(data.as_of) : null;
  var catalog = (store.settings && store.settings.catalogURL) || '';
  var showSource = /^https:\/\//i.test(catalog);
  var src = catalog;
  return '<div class="dash-foot">' +
    '<span class="dot ' + (stale ? 'amber' : 'green') + '"></span>' +
    '<span>' + (stale ? 'Last loaded data' : 'Backend data') + '</span>' +
    (asOf ? '<span class="time">' + esc(formatTime(asOf.getTime() / 1000)) + '</span>' : '') +
    (stale ? '<button class="info-btn" title="' + esc(claimStatus) + '" aria-label="Some backend data may be delayed">' +
      icon('info', 13) + '</button>' : '') +
    '<span class="spacer"></span>' +
    (showSource && src ? '<a href="' + attr(src) + '" target="_blank" rel="noreferrer">View source</a>' : '') +
  '</div>';
}

function parseISO(text) {
  if (!text) return null;
  var d = new Date(text);
  if (isNaN(d.getTime())) {
    // some feeds use fractional-second ISO that older parsers reject
    var m = String(text).match(/^(\d{4}-\d{2}-\d{2}T[\d:]+)(\.\d+)?/);
    if (m) d = new Date(m[1]);
  }
  return isNaN(d.getTime()) ? null : d;
}

// ===== Chart (hand-rolled SVG) ==============================================

function normIntervals(points) {
  var out = [];
  (points || []).forEach(function (p) {
    var from, to, value;
    if (Array.isArray(p)) { from = p[0]; to = p.length > 2 ? p[2] : p[0]; value = p[1]; }
    else if (p && typeof p === 'object') {
      from = (p.from != null) ? p.from : (p.x != null ? p.x : p.start);
      to = (p.to != null) ? p.to : (p.end != null ? p.end : from);
      value = (p.value != null) ? p.value : (p.y != null ? p.y : p.v);
    } else return;
    from = Number(from); to = Number(to); value = Number(value);
    if (!isFinite(from) || !isFinite(value)) return;
    if (!isFinite(to) || to < from) to = from;
    out.push({ from: from, to: to, value: value });
  });
  out.sort(function (a, b) { return a.from - b.from; });
  return out;
}

function niceTicks(min, max, count) {
  if (!isFinite(min) || !isFinite(max)) return [];
  if (max <= min) max = min + 1;
  var span = max - min;
  var mag = Math.pow(10, Math.floor(Math.log(span / Math.max(1, count)) / Math.LN10));
  var candidates = [1, 2, 5, 10];
  var step = 10 * mag;
  for (var i = 0; i < candidates.length; i++) {
    var s = candidates[i] * mag;
    if (Math.floor(span / s) + 1 <= count + 1) { step = s; break; }
  }
  var start = Math.ceil(min / step) * step;
  var ticks = [];
  var n = Math.floor((max - start) / step + 1e-9);
  for (var j = 0; j <= n; j++) ticks.push(Number((start + j * step).toFixed(10)));
  return ticks;
}

function xAxisTicks(minMs, maxMs) {
  var day = 86400000;
  var span = Math.max(day, maxMs - minMs);
  var steps = [1, 2, 3, 7, 14, 30, 60, 90, 180, 365, 730, 3650];
  var stepDays = steps[steps.length - 1];
  for (var i = 0; i < steps.length; i++) {
    if (Math.floor(span / (steps[i] * day)) <= 3) { stepDays = steps[i]; break; }
  }
  var stepMs = stepDays * day;
  var n = Math.floor(span / stepMs);
  var offset = (span - n * stepMs) / 2; // center the ticks across the range
  var ticks = [];
  for (var k = 0; k <= n; k++) ticks.push(minMs + offset + k * stepMs);
  return { ticks: ticks, stepDays: stepDays };
}

function renderChart() {
  var project = selectedProject();
  var entry = project ? getDashEntry(project.id) : null;
  var host = $('#chartHost');
  if (!host || !entry || !entry.data) return;
  var groups = entry.data.groups || [];
  var group = groups.find(function (g) { return g.id === entry.groupID; }) || groups[0];
  if (!group) return;
  var hist = entry.observed ? group.observed : group.estimated;
  var intervals = normIntervals(hist ? hist.points : []);
  if (!intervals.length) return;

  var W = Math.max(320, host.clientWidth || 680);
  var H = 245;
  var padL = 38, padR = 10, padT = 10, padB = 24;
  var plotW = W - padL - padR;
  var plotH = H - padT - padB;

  var xMin = intervals[0].from;
  var xMax = Math.max.apply(null, intervals.map(function (iv) { return iv.to; }));
  if (xMax < intervals[intervals.length - 1].to) xMax = intervals[intervals.length - 1].to;
  if (xMax <= xMin) xMax = xMin + daySpanFallback(intervals);

  var yMin = Math.min.apply(null, intervals.map(function (iv) { return iv.value; }));
  var yMax = Math.max.apply(null, intervals.map(function (iv) { return iv.value; }));
  if (yMax <= yMin) { yMax += 1; yMin -= 1; }
  var yPad = (yMax - yMin) * 0.1;
  var yLo = yMin - yPad, yHi = yMax + yPad;

  var xTickInfo = xAxisTicks(xMin * 1000, xMax * 1000);
  var yTicks = niceTicks(yLo, yHi, 4);

  function X(t) { return padL + ((t - xMin) / (xMax - xMin)) * plotW; }
  function Y(v) { return padT + (1 - (v - yLo) / (yHi - yLo)) * plotH; }

  var svg = ['<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Metric history chart">'];

  // horizontal gridlines + y labels
  yTicks.forEach(function (t) {
    var y = Y(t);
    if (y < padT - 1 || y > padT + plotH + 1) return;
    svg.push('<line x1="' + padL + '" y1="' + y.toFixed(1) + '" x2="' + (W - padR) + '" y2="' + y.toFixed(1) +
      '" stroke="rgba(0,0,0,0.06)" stroke-width="1"/>');
    svg.push('<text x="' + (padL - 7) + '" y="' + (y + 3.5).toFixed(1) + '" text-anchor="end" font-size="10"' +
      ' fill="#75787a" font-variant-numeric="tabular-nums">' + esc(formatMetricNumber(t)) + '</text>');
  });

  // dashed vertical gridlines at date ticks + x labels
  var includeYear = xTickInfo.stepDays >= 365;
  xTickInfo.ticks.forEach(function (t) {
    var x = X(t / 1000);
    if (x < padL - 1 || x > W - padR + 1) return;
    svg.push('<line x1="' + x.toFixed(1) + '" y1="' + padT + '" x2="' + x.toFixed(1) + '" y2="' + (padT + plotH) +
      '" stroke="rgba(0,0,0,0.14)" stroke-width="1" stroke-dasharray="2 5"/>');
    svg.push('<text x="' + x.toFixed(1) + '" y="' + (H - 7) + '" text-anchor="middle" font-size="10" fill="#75787a">' +
      esc(xLabel(new Date(t), includeYear)) + '</text>');
  });

  // step line: hold each value until the next interval starts, then drop; last extends to its `to`
  var d = '';
  intervals.forEach(function (iv, i) {
    var x0 = X(iv.from), y0 = Y(iv.value);
    if (i === 0) d += 'M' + x0.toFixed(1) + ',' + y0.toFixed(1);
    var isLast = i === intervals.length - 1;
    var endT = isLast ? iv.to : intervals[i + 1].from;
    var x1 = X(endT);
    d += 'L' + x1.toFixed(1) + ',' + y0.toFixed(1);
    if (!isLast) {
      var y1 = Y(intervals[i + 1].value);
      d += 'L' + x1.toFixed(1) + ',' + y1.toFixed(1);
    }
  });
  svg.push('<path d="' + d + '" fill="none" stroke="#4d57c7" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>');
  svg.push('</svg>');
  host.innerHTML = svg.join('');
}

function daySpanFallback(intervals) {
  var span = 0;
  for (var i = 0; i < intervals.length; i++) span = Math.max(span, intervals[i].to - intervals[i].from);
  return span > 0 ? span : 86400;
}

function xLabel(date, includeYear) {
  if (includeYear) return date.toLocaleDateString(undefined, { year: 'numeric' });
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

// ===== Sheets ===============================================================

var sheetEl = null;

function closeSheet() {
  if (sheetEl) { sheetEl.remove(); sheetEl = null; }
  store.loginProvider = null;
  store.loginOutput = '';
  store.loginSpawn = null;
}

function sheetFrame(label, bodyId, footId) {
  var backdrop = document.createElement('div');
  backdrop.className = 'sheet-backdrop';
  backdrop.innerHTML =
    '<div class="sheet sheet-settings" role="dialog" aria-label="' + attr(label) + '">' +
      '<div class="sheet-head"><span class="sheet-title">' + esc(label) + '</span><span class="spacer"></span>' +
        '<button class="sheet-done" id="settingsDone">Done</button></div>' +
      '<div class="sheet-body" id="' + bodyId + '"></div>' +
      '<div class="sheet-foot" id="' + footId + '">' +
        '<span class="sheet-note">Changes merge into Runtime/settings.json.</span><span class="spacer"></span>' +
        '<button class="sheet-primary" id="settingsSave">Save</button>' +
      '</div>' +
    '</div>';
  $('#sheetHost').appendChild(backdrop);
  backdrop.addEventListener('mousedown', function (e) {
    if (e.target === backdrop) closeSheet();
  });
  return backdrop;
}

var SETTINGS_TABS = ['Connections', 'Cloud', 'Lean server', 'MCP tools', 'Research policy', 'Workspace'];

function openSettingsSheet() {
  if (sheetEl) return;
  closeMenu();
  sheetEl = sheetFrame('Your research workspace', 'settingsBody', 'settingsFoot');
  $('#settingsDone', sheetEl).addEventListener('click', function () {
    saveSettings().then(function () { closeSheet(); }).catch(function () { /* keep open */ });
  });
  $('#settingsSave', sheetEl).addEventListener('click', function () { saveSettings(); });
  buildSettingsBody();
  loadSettingsDraft();
  loadPolicyDraft();
  loadMcpServers();
  apiGet('/healthz').then(function (res) {
    store.health = res;
    renderServerBox();
  }).catch(function () { /* health is optional */ });
}

function showSettingsTab(name) {
  if (!sheetEl) return;
  store.settingsTab = name;
  $all('.seg-tabs button', sheetEl).forEach(function (b) {
    var active = b.getAttribute('data-tab') === name;
    b.classList.toggle('active', active);
    b.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
  $all('[data-pane]', sheetEl).forEach(function (pane) {
    pane.hidden = pane.getAttribute('data-pane') !== name;
  });
}

function buildSettingsBody() {
  var body = $('#settingsBody', sheetEl);
  if (!body) return;
  var tabs = '<div class="seg-tabs" role="tablist" aria-label="Settings">' +
    SETTINGS_TABS.map(function (t) {
      return '<button role="tab" data-tab="' + t + '">' + esc(t) + '</button>';
    }).join('') + '</div>';
  var panes = [];

  // Connections — the provider cards scroll, like the native sheet.
  var cards = PROVIDERS.map(function (p) { return providerCardHTML(p); }).join('');
  panes.push('<div data-pane="Connections">' +
    '<div class="pane-scroll">' + cards + '</div>' +
    '<p class="section-note">Sign-in uses the official CLI and your browser. Credentials stay with each CLI; the server does not copy or store them. Permission errors remain visible in chat.</p>' +
    '<button class="sheet-secondary" id="refreshAllAuth">Refresh connection status</button>' +
    '<div id="loginBlock"></div>' +
  '</div>');

  // Cloud — 1:1 with the native Cloud tab (ConnectionsView.swift).
  panes.push('<div data-pane="Cloud">' +
    '<div class="field-row"><span class="field-label">Run research</span>' +
      '<div class="segmented run-picker" id="cloudRunPicker">' +
        '<button data-target="mac" aria-pressed="true">This device</button>' +
        '<button data-target="cloud" aria-pressed="false"><span id="cloudRunCloudLabel">Daytona sandboxes · Codex</span></button>' +
      '</div></div>' +
    '<p class="section-note" id="cloudRunNote" hidden></p>' +
    '<div class="field"><label for="cloudProvider">Sandbox provider</label>' +
      '<select id="cloudProvider"><option value="daytona">Daytona</option><option value="agent37">Agent37</option></select></div>' +
    '<h3 class="pane-heading">Private parallel sandboxes</h3>' +
    '<p class="section-note">Each agent has its own Linux checkout. Share context uses live-written checkpoint files, combines them directly, then supplies the bundle to fresh sessions. Your account files are never uploaded.</p>' +
    '<div class="field-row">' +
      '<div class="field"><label for="cloudKey" id="cloudKeyLabel">Daytona API key</label>' +
        '<input id="cloudKey" type="password" placeholder="Daytona API key" autocomplete="off"></div>' +
      '<div class="field narrow"><button class="sheet-secondary" id="cloudSaveKey">Save key</button></div>' +
    '</div>' +
    '<p class="section-note" id="cloudStatus">Not connected</p>' +
    '<div class="field-row">' +
      '<button class="sheet-secondary" id="cloudConnect">Connect</button>' +
      '<button class="sheet-secondary" id="cloudSignIn">Sign in to cloud Codex</button>' +
    '</div>' +
    '<p class="section-note" id="cloudNote"></p>' +
    '<p class="section-note" id="cloudLoginNote" hidden></p>' +
    '<div class="field-row">' +
      '<button class="sheet-secondary" id="cloudStop">Stop owned cloud compute</button>' +
      '<button class="sheet-secondary" id="cloudRemoveKey">Remove saved key</button>' +
    '</div>' +
    '<p class="section-note" id="cloudBusyNote" hidden></p>' +
  '</div>');

  panes.push('<div data-pane="Lean server">' +
    '<h3 class="pane-heading">Shared Lean verification</h3>' +
    '<p class="section-note">Search on your agents; compile and verify on this device. The server retains its dependency cache, candidate sources, failed checks and receipts.</p>' +
    '<div class="field"><label for="leanStorage">Server folder</label>' +
      '<input id="leanStorage" placeholder="A dedicated folder outside Documents, Desktop and iCloud" spellcheck="false"></div>' +
    '<div class="field"><label for="leanPath">Installed Lean executable</label>' +
      '<input id="leanPath" placeholder="Installed Lean executable" spellcheck="false"></div>' +
    '<div class="field-row">' +
      '<button class="sheet-secondary" id="leanStart">Start server</button>' +
      '<button class="sheet-secondary" id="leanStop">Stop server</button>' +
      '<button class="sheet-secondary" id="leanOpenLogs">Open profiles and logs</button>' +
    '</div>' +
    '<p class="section-note" id="leanStatus">Stopped</p>' +
    '<p class="section-note" id="leanLocalURL" hidden></p>' +
    '<p class="section-note">Local research runs receive the connection automatically. The initial kernel-check profile tests the service; full QRH verification needs its prepared proof project and pinned challenge in server-config.json. Stop the server before editing profiles.</p>' +
    '<h3 class="pane-heading">Allow other nodes to connect</h3>' +
    '<div class="field"><label for="leanNgrok">ngrok executable</label>' +
      '<input id="leanNgrok" placeholder="ngrok" spellcheck="false"></div>' +
    '<div class="field-row">' +
      '<button class="sheet-secondary" id="leanShare">Share with ngrok</button>' +
      '<button class="sheet-secondary" id="leanStopShare">Stop sharing</button>' +
      '<button class="sheet-secondary" id="leanCopyToken">Copy access token</button>' +
    '</div>' +
    '<p class="section-note" id="leanPublicURL" hidden></p>' +
    '<p class="section-note">Uses your existing ngrok login. Share the HTTPS address and access token only with intended nodes. Requests are authenticated; submitted source runs in a restricted process. Stopping sharing closes public access while local verification stays available.</p>' +
  '</div>');

  panes.push('<div data-pane="MCP tools">' +
    '<p class="section-note">Choose which direct Codex MCP connections this app may start. None are enabled by default. CLI plugins are not loaded in research runs. Your global configuration is preserved.</p>' +
    '<div id="mcpList" class="mcp-list"><div class="d-detail">Loading…</div></div>' +
    '<button class="sheet-secondary" id="refreshMcp">Refresh available servers</button>' +
  '</div>');

  panes.push('<div data-pane="Research policy">' +
    '<p class="section-note">These instructions guide both models. Record failures and expected behavior here to improve future runs.</p>' +
    '<textarea id="policyEditor" class="mono-editor" rows="14" spellcheck="false" aria-label="Research policy"></textarea>' +
  '</div>');

  panes.push('<div data-pane="Workspace">' +
    '<h3 class="pane-heading">External SSD workspace</h3>' +
    '<div class="field"><label for="setWorkspaceRoot">Workspace root</label>' +
      '<input id="setWorkspaceRoot" data-key="workspaceRoot" placeholder="Projects, chats, logs and caches"></div>' +
    '<p class="section-note">Projects, chat history, raw run logs, caches and temporary files stay here. The app stops new writes if the volume disappears. Your CLI keeps its own account configuration in its normal location.</p>' +
    '<button class="sheet-secondary" id="showWorkspaceFolder">Show workspace folder</button>' +
    '<h3 class="pane-heading">Live research catalog</h3>' +
    '<div class="field"><label for="setCatalogURL">HTTPS catalog URL</label>' +
      '<input id="setCatalogURL" data-key="catalogURL" placeholder="https://…/auto-research"></div>' +
    '<p class="section-note">The catalog is refreshed on launch and when you press Refresh in the sidebar. Your existing chats are preserved. New repositories can also be added manually.</p>' +
    '<div class="section-title">Dashboard server</div><div id="serverBox"></div>' +
  '</div>');

  body.innerHTML = tabs + panes.join('');
  $all('.seg-tabs button', sheetEl).forEach(function (b) {
    b.addEventListener('click', function () { showSettingsTab(b.getAttribute('data-tab')); });
  });
  var refreshAuthBtn = $('#refreshAllAuth', sheetEl);
  if (refreshAuthBtn) refreshAuthBtn.addEventListener('click', function () {
    apiGet('/api/auth/status').then(function (res) {
      store.auth = Object.assign({}, store.auth, (res && res.auth) || {});
      if (store.settingsDraft) store.settingsDraft.auth = Object.assign({}, store.settingsDraft.auth, store.auth);
      updateSettingsSheetAuth();
      toast('Connection status refreshed');
    }).catch(apiFail);
  });
  var refreshMcp = $('#refreshMcp', sheetEl);
  if (refreshMcp) refreshMcp.addEventListener('click', function () {
    refreshMcp.textContent = 'Refreshing…';
    apiPost('/api/mcp/refresh', {}).then(function (res) {
      renderMcpList(res);
      refreshMcp.textContent = 'Refresh available servers';
    }).catch(function (err) {
      refreshMcp.textContent = 'Refresh available servers';
      apiFail(err);
    });
  });
  wireProviderCards();
  fillSettingsValues(store.settingsDraft);
  renderServerBox();
  wireCloudPane();
  wireLeanPane();
  loadCloudStatus();
  loadLeanSettings();
  var workspaceBtn = $('#showWorkspaceFolder', sheetEl);
  if (workspaceBtn) workspaceBtn.addEventListener('click', function () {
    apiPost('/api/workspace/open', {}).then(function () {
      toast('Opened the workspace folder');
    }).catch(apiFail);
  });
  var policyEditor = $('#policyEditor', sheetEl);
  if (policyEditor) policyEditor.addEventListener('input', function () { policyEditor.dirty = true; });
  showSettingsTab(store.settingsTab || 'Connections');
}

function draftSettings() {
  return (store.settingsDraft && store.settingsDraft.settings) || store.settings || {};
}

function detectedPaths() {
  return (store.settingsDraft && store.settingsDraft.detectedPaths) || {};
}

function loadSettingsDraft() {
  apiGet('/api/settings').then(function (res) {
    if (!res || typeof res !== 'object') return;
    var draft = {
      settings: Object.assign({}, store.settings, res.settings || res),
      detectedPaths: res.detectedPaths || {},
      auth: res.auth || store.auth
    };
    store.settingsDraft = draft;
    if (sheetEl) fillSettingsValues(draft);
  }).catch(function (err) {
    if (err && err.status === 401) { promptToken(); return; }
    /* fall back to the state copy; the fields still show */
  });
}

var policyDraft = '';

function loadPolicyDraft() {
  apiGet('/api/policy').then(function (res) {
    policyDraft = (res && res.text) || '';
    var editor = sheetEl && $('#policyEditor', sheetEl);
    if (editor && !editor.dirty) editor.value = policyDraft;
  }).catch(function (err) {
    if (err && err.status === 401) { promptToken(); return; }
  });
}

var mcpDraft = { available: [], enabled: [] };

function loadMcpServers() {
  apiGet('/api/mcp').then(function (res) { renderMcpList(res); }).catch(function () { /* optional */ });
}

function renderMcpList(res) {
  mcpDraft = { available: (res && res.available) || [], enabled: (res && res.enabled) || [] };
  if (!sheetEl) return;
  var list = $('#mcpList', sheetEl);
  if (!list) return;
  if (!mcpDraft.available.length) {
    list.innerHTML = '<div class="d-detail">No direct Codex MCP connections were found. Install and configure servers in the Codex CLI, then refresh.</div>';
    return;
  }
  list.innerHTML = mcpDraft.available.map(function (name) {
    var on = mcpDraft.enabled.indexOf(name) !== -1;
    return '<label class="mcp-row"><input type="checkbox" data-mcp="' + attr(name) + '"' +
      (on ? ' checked' : '') + '><span>' + esc(name) + '</span></label>';
  }).join('');
  $all('[data-mcp]', list).forEach(function (box) {
    box.addEventListener('change', function () {
      var name = box.getAttribute('data-mcp');
      if (box.checked) {
        if (mcpDraft.enabled.indexOf(name) === -1) mcpDraft.enabled.push(name);
      } else {
        mcpDraft.enabled = mcpDraft.enabled.filter(function (n) { return n !== name; });
      }
    });
  });
}

var PERMS = {
  codex: [{ key: 'codexSandbox', label: 'Access', options: [['workspace-write', 'Workspace writes'], ['read-only', 'Read only']] }],
  claude: [{ key: 'claudePermissionMode', label: 'Permissions', options: [['auto', 'Automatic review'], ['acceptEdits', 'Accept edits'], ['plan', 'Plan only'], ['dontAsk', 'Deny unapproved']] }],
  opencode: [{ key: 'opencodePermission', label: 'Permissions', options: [['reject', 'Reject prompts'], ['auto', 'Auto-approve']] }],
  antigravity: [
    { key: 'antigravityAccess', label: 'Access', options: [['accept-edits', 'Accept edits'], ['auto', 'Auto'], ['plan', 'Plan only'], ['unrestricted', 'Auto-approve all']] }
  ]
};

function providerCardHTML(p) {
  var title = PROVIDER_TITLES[p];
  var perms = PERMS[p];
  var permHTML = perms.map(function (perm) {
    return '<div class="field"><label>' + esc(perm.label) + '</label>' +
      '<select data-key="' + perm.key + '">' +
      perm.options.map(function (o) {
        return '<option value="' + o[0] + '">' + esc(o[1]) + '</option>';
      }).join('') + '</select></div>';
  }).join('');
  var modelPlaceholder = p === 'opencode'
    ? 'Model as provider/model (empty uses your CLI default)' : 'Model (empty uses your CLI default)';
  return '<div class="provider-card">' +
    '<div class="provider-head">' +
      '<span class="p-icon">' + icon('terminal', 20) + '</span>' +
      '<div><div class="provider-name">' + esc(title) + '</div>' +
      '<div class="provider-status" id="pstat-' + p + '">' + esc(authStatus(p)) + '</div></div>' +
      '<span class="spacer"></span>' +
      '<button class="p-btn" data-signin="' + p + '">Sign in</button>' +
      '<a class="p-btn" href="' + INSTALL_URLS[p] + '" target="_blank" rel="noreferrer">Install CLI</a>' +
    '</div>' +
    '<div class="field"><label for="path-' + p + '">CLI executable path</label>' +
      '<input id="path-' + p + '" data-key="' + p + 'Path" placeholder="CLI executable path"></div>' +
    '<div class="field-row">' +
      '<div class="field"><label for="model-' + p + '">Model</label>' +
        '<input id="model-' + p + '" data-key="' + p + 'Model" placeholder="' + esc(modelPlaceholder) + '"></div>' +
      '<div class="field narrow">' + permHTML + '</div>' +
    '</div>' +
  '</div>';
}

function wireProviderCards() {
  $all('[data-refresh]', sheetEl).forEach(function (b) {
    b.addEventListener('click', function () { refreshAccount(b.getAttribute('data-refresh')); });
  });
  $all('[data-signin]', sheetEl).forEach(function (b) {
    b.addEventListener('click', function () { signIn(b.getAttribute('data-signin')); });
  });
}

function fillSettingsValues(draft) {
  if (!sheetEl) return;
  var settings = draftSettings();
  var detected = detectedPaths();
  $all('[data-key]', sheetEl).forEach(function (input) {
    var key = input.getAttribute('data-key');
    if (key === 'antigravityAccess') {
      input.value = (settings.antigravityPermission === 'auto' ? 'unrestricted'
        : settings.antigravityMode === 'plan' ? 'plan'
        : settings.antigravityMode === 'auto' ? 'auto' : 'accept-edits');
      return;
    }
    if (key.indexOf('Path') !== -1) {
      var p = key.replace(/Path$/, '');
      var placeholder = detected[p] || 'CLI executable path';
      input.setAttribute('placeholder', placeholder);
    }
    if (settings[key] != null) input.value = settings[key];
  });
  $all('#pstat-codex, #pstat-claude, #pstat-opencode, #pstat-antigravity', sheetEl).forEach(function (el) {
    var p = el.id.replace('pstat-', '');
    el.textContent = (store.settingsDraft && store.settingsDraft.auth && store.settingsDraft.auth[p]) || authStatus(p);
  });
}

function updateSettingsSheetAuth() {
  if (!sheetEl) return;
  var draft = store.settingsDraft;
  PROVIDERS.forEach(function (p) {
    var el = $('#pstat-' + p, sheetEl);
    if (el) el.textContent = (draft && draft.auth && draft.auth[p]) || authStatus(p);
  });
}

function refreshAccount(provider) {
  apiGet('/api/auth/status?provider=' + encodeURIComponent(provider)).then(function (res) {
    var status = null;
    if (typeof res === 'string') status = res;
    else if (res && typeof res === 'object') {
      status = res.status || res.state || (res.auth ? res.auth[provider] : res[provider]);
      if (res.auth && typeof res.auth === 'object') {
        store.auth = Object.assign({}, store.auth, res.auth);
        if (store.settingsDraft) store.settingsDraft.auth = Object.assign({}, store.settingsDraft.auth, res.auth);
      } else if (status) {
        store.auth[provider] = status;
        if (!store.settingsDraft) {
          store.settingsDraft = { settings: Object.assign({}, store.settings), detectedPaths: {}, auth: Object.assign({}, store.auth) };
        }
        store.settingsDraft.auth[provider] = status;
      }
    }
    updateSettingsSheetAuth();
    renderHeader();
    if (status) toast(PROVIDER_TITLES[provider] + ': ' + status);
  }).catch(function (err) { apiFail(err); });
}

function signIn(provider) {
  apiPost('/api/auth/login', { provider: provider }).then(function (res) {
    store.loginProvider = provider;
    if (res && res.spawned) {
      store.loginOutput = null;
      store.loginSpawn = res;
      renderLoginBlock();
      return;
    }
    var text = typeof res === 'string' ? res :
      (res && (res.output || res.instructions || res.text || res.message)) ? String(res.output || res.instructions || res.text || res.message)
      : JSON.stringify(res, null, 2);
    store.loginSpawn = null;
    store.loginOutput = text;
    renderLoginBlock();
  }).catch(function (err) { apiFail(err); });
}

function signInInServer(provider) {
  apiPost('/api/auth/login', { provider: provider, spawn: false }).then(function (res) {
    var text = typeof res === 'string' ? res :
      (res && (res.output || res.instructions || res.text || res.message || res.instructions)) ?
      String(res.output || res.instructions || res.text || res.message) : JSON.stringify(res, null, 2);
    store.loginProvider = provider;
    store.loginSpawn = null;
    store.loginOutput = text;
    renderLoginBlock();
  }).catch(function (err) { apiFail(err); });
}

function openLoginNote(res) {
  store.loginProvider = 'okc';
  store.loginSpawn = res || {};
  store.loginOutput = null;
  renderLoginBlock();
  if (sheetEl) showSettingsTab('Connections');
}

function renderLoginBlock() {
  if (!sheetEl) return;
  var block = $('#loginBlock', sheetEl);
  if (!block) return;
  if (!store.loginProvider) { block.innerHTML = ''; return; }
  var head = '<div class="login-head"><span>' + esc(PROVIDER_TITLES[store.loginProvider] || 'OKC') + ' sign-in</span><span class="spacer"></span>' +
    '<button class="sheet-secondary" id="closeLogin">Close</button></div>';
  var spawn = store.loginSpawn;
  if (spawn && store.loginOutput == null) {
    block.innerHTML = head +
      '<div class="login-output">' +
        '<p>' + esc(spawn.note || 'A terminal window opened on the device that serves this dashboard.') + '</p>' +
        (spawn.spawned ? '<p class="section-note">Complete the sign-in there, then press Refresh account. The window never shares credentials with the server.</p>' : '') +
        '<p class="mono-note">' + esc(spawn.script || '') + '</p>' +
      '</div>' +
      '<div class="login-actions"><button class="sheet-secondary" id="loginInServer">Run in the server session instead</button></div>';
  } else {
    block.innerHTML = head +
      '<div class="login-output">' + esc(store.loginOutput || '') + '</div>' +
      '<div class="login-actions"><button class="sheet-secondary" id="loginSpawnHost">Open a terminal on the host device</button></div>';
  }
  $('#closeLogin', block).addEventListener('click', function () {
    store.loginProvider = null;
    store.loginOutput = '';
    store.loginSpawn = null;
    renderLoginBlock();
  });
  var inServer = $('#loginInServer', block);
  if (inServer) inServer.addEventListener('click', function () {
    signInInServer(store.loginProvider === 'okc' ? 'opencode' : store.loginProvider);
  });
  var spawnHost = $('#loginSpawnHost', block);
  if (spawnHost) spawnHost.addEventListener('click', function () {
    var provider = store.loginProvider === 'okc' ? 'opencode' : store.loginProvider;
    signIn(provider);
  });
}

// ===== Cloud pane (the native Cloud tab) ====================================

var CLOUD_TITLES = { daytona: 'Daytona', agent37: 'Agent37' };

function cloudNoteText(provider) {
  if (provider === 'agent37') {
    return 'The API key is stored in this workspace. Cloud model access uses the official device login separately. Cloud compute is billed to your Agent37 account. Completed work is downloaded and verified before workers are stopped with their disks retained; failed exports preserve the sandbox. Idle sandboxes sleep after 60 minutes.';
  }
  return 'The API key is stored in this workspace. Cloud model access uses the official device login separately. Cloud compute is billed to your Daytona account. Completed work is downloaded and verified before workers are archived; failed exports preserve the sandbox. Idle sandboxes stop after 60 minutes.';
}

function renderCloudPane() {
  if (!sheetEl) return;
  var cloud = store.cloud || {};
  var title = CLOUD_TITLES[cloud.provider] || 'Daytona';
  var cloudLabel = $('#cloudRunCloudLabel', sheetEl);
  if (cloudLabel) cloudLabel.textContent = title + ' sandboxes · Codex';
  var macBtn = $('#cloudRunPicker [data-target="mac"]', sheetEl);
  var cloudBtn = $('#cloudRunPicker [data-target="cloud"]', sheetEl);
  if (macBtn) {
    macBtn.classList.toggle('active', cloud.runTarget !== 'cloud');
    macBtn.setAttribute('aria-pressed', cloud.runTarget === 'cloud' ? 'false' : 'true');
  }
  if (cloudBtn) {
    cloudBtn.classList.toggle('active', cloud.runTarget === 'cloud');
    cloudBtn.setAttribute('aria-pressed', cloud.runTarget === 'cloud' ? 'true' : 'false');
  }
  var runNote = $('#cloudRunNote', sheetEl);
  if (runNote) {
    runNote.hidden = cloud.runTarget !== 'cloud';
    runNote.textContent = 'Cloud sandbox execution needs the desktop edition\u2019s sandbox '
      + 'bridge, arriving in a later release. Play stays on this device in this edition; '
      + 'Connect still validates the saved key and lists the compute this installation owns.';
  }
  var select = $('#cloudProvider', sheetEl);
  if (select) select.value = cloud.provider || 'daytona';
  var keyLabel = $('#cloudKeyLabel', sheetEl);
  if (keyLabel) keyLabel.textContent = title + ' API key';
  var keyInput = $('#cloudKey', sheetEl);
  if (keyInput) keyInput.placeholder = title + ' API key';
  var status = $('#cloudStatus', sheetEl);
  if (status) status.textContent = (cloud.status && cloud.status[cloud.provider]) || 'Not connected';
  var note = $('#cloudNote', sheetEl);
  if (note) note.textContent = cloudNoteText(cloud.provider || 'daytona');
  var busy = !!cloud.busy;
  ['cloudConnect', 'cloudSignIn', 'cloudStop', 'cloudRemoveKey', 'cloudSaveKey'].forEach(function (id) {
    var button = $('#' + id, sheetEl);
    if (button) button.disabled = busy;
  });
  var busyNote = $('#cloudBusyNote', sheetEl);
  if (busyNote) {
    busyNote.hidden = !busy;
    busyNote.textContent = busy === 'stopping'
      ? 'Verifying and stopping this installation\u2019s compute…'
      : 'Preparing private cloud connection…';
  }
}

function loadCloudStatus() {
  apiGet('/api/cloud/status').then(function (res) {
    if (res) store.cloud = Object.assign({}, store.cloud, res);
    renderCloudPane();
  }).catch(function () { renderCloudPane(); });
}

function cloudSaveKey() {
  var input = $('#cloudKey', sheetEl);
  var key = (input && input.value || '').trim();
  var title = CLOUD_TITLES[store.cloud.provider] || 'Daytona';
  if (!key) { toast('Enter a ' + title + ' API key first.'); return; }
  apiPost('/api/cloud/key', { provider: store.cloud.provider, key: key }).then(function (res) {
    if (input) input.value = '';
    store.cloud = Object.assign({}, store.cloud, res);
    renderCloudPane();
  }).catch(apiFail);
}

function cloudConnect() {
  store.cloud.busy = 'connecting';
  renderCloudPane();
  apiPost('/api/cloud/connect', { provider: store.cloud.provider }).then(function (res) {
    store.cloud = Object.assign({}, store.cloud, res);
    renderCloudPane();
  }).catch(function (err) {
    store.cloud.busy = '';
    renderCloudPane();
    apiFail(err);
  });
}

function cloudStop() {
  store.cloud.busy = 'stopping';
  renderCloudPane();
  apiPost('/api/cloud/stop', { provider: store.cloud.provider }).then(function (res) {
    store.cloud = Object.assign({}, store.cloud, res);
    renderCloudPane();
  }).catch(function (err) {
    store.cloud.busy = '';
    renderCloudPane();
    apiFail(err);
  });
}

function cloudRemoveKey() {
  apiPost('/api/cloud/remove-key', { provider: store.cloud.provider }).then(function (res) {
    store.cloud = Object.assign({}, store.cloud, res);
    renderCloudPane();
  }).catch(apiFail);
}

function cloudSignIn() {
  apiPost('/api/cloud/codex-login', { provider: store.cloud.provider }).then(function (res) {
    var block = $('#cloudLoginNote', sheetEl);
    if (block) {
      block.hidden = false;
      block.innerHTML = esc((res && res.note)
        || 'A terminal window opened on the device that serves this dashboard.')
        + ' Complete the device login there, or open <a href="https://auth.openai.com/codex/device"'
        + ' target="_blank" rel="noreferrer">Codex device sign-in</a>.';
    }
  }).catch(apiFail);
}

function wireCloudPane() {
  if (!sheetEl) return;
  var picker = $('#cloudRunPicker', sheetEl);
  if (picker) $all('button', picker).forEach(function (button) {
    button.addEventListener('click', function () {
      var target = button.getAttribute('data-target');
      if (store.cloud.runTarget === target) return;
      apiPost('/api/cloud/run-target', { target: target }).then(function (res) {
        store.cloud = Object.assign({}, store.cloud, res);
        renderCloudPane();
      }).catch(apiFail);
    });
  });
  var select = $('#cloudProvider', sheetEl);
  if (select) select.addEventListener('change', function () {
    apiPost('/api/cloud/provider', { provider: select.value }).then(function (res) {
      store.cloud = Object.assign({}, store.cloud, res);
      var keyInput = $('#cloudKey', sheetEl);
      if (keyInput) keyInput.value = '';
      renderCloudPane();
    }).catch(apiFail);
  });
  var pairs = { cloudSaveKey: cloudSaveKey, cloudConnect: cloudConnect,
                cloudSignIn: cloudSignIn, cloudStop: cloudStop,
                cloudRemoveKey: cloudRemoveKey };
  Object.keys(pairs).forEach(function (id) {
    var button = $('#' + id, sheetEl);
    if (button) button.addEventListener('click', pairs[id]);
  });
  renderCloudPane();
}

// ===== Lean server pane =====================================================

function renderLeanPane() {
  if (!sheetEl) return;
  var lean = store.lean || {};
  var status = $('#leanStatus', sheetEl);
  if (status) status.textContent = lean.status || 'Stopped';
  var local = $('#leanLocalURL', sheetEl);
  if (local) { local.hidden = !lean.localURL; local.textContent = lean.localURL || ''; }
  var pub = $('#leanPublicURL', sheetEl);
  if (pub) { pub.hidden = !lean.publicURL; pub.textContent = lean.publicURL || ''; }
  var busy = !!lean.busy;
  ['leanStart', 'leanStop', 'leanOpenLogs', 'leanShare', 'leanStopShare', 'leanCopyToken']
    .forEach(function (id) {
      var button = $('#' + id, sheetEl);
      if (button) button.disabled = busy;
    });
}

function fillLeanFields() {
  if (!sheetEl) return;
  var lean = store.lean || {};
  var storage = $('#leanStorage', sheetEl);
  if (storage) storage.value = lean.storagePath || '';
  var path = $('#leanPath', sheetEl);
  if (path) path.value = lean.leanPath || '';
  var ngrok = $('#leanNgrok', sheetEl);
  if (ngrok) ngrok.value = lean.ngrokPath || '';
}

function leanDraftBody() {
  var storage = $('#leanStorage', sheetEl), path = $('#leanPath', sheetEl),
      ngrok = $('#leanNgrok', sheetEl);
  return { storagePath: storage ? storage.value : '', leanPath: path ? path.value : '',
           ngrokPath: ngrok ? ngrok.value : '' };
}

function leanSave() {
  return apiPost('/api/lean/settings', leanDraftBody()).then(function (res) {
    if (res) store.lean = Object.assign({}, store.lean, res);
    renderLeanPane();
    return null;
  });
}

function loadLeanSettings() {
  apiGet('/api/lean/settings').then(function (res) {
    if (res) store.lean = Object.assign({}, store.lean, res);
    fillLeanFields();
    renderLeanPane();
  }).catch(function () { renderLeanPane(); });
}

function leanStart() {
  leanSave().then(function () {
    apiPost('/api/lean/start', {}).then(function (res) {
      store.lean = Object.assign({}, store.lean, res);
      renderLeanPane();
    }).catch(function (err) { loadLeanSettings(); apiFail(err); });
  }).catch(apiFail);
}

function leanStop() {
  apiPost('/api/lean/stop', {}).then(function (res) {
    store.lean = Object.assign({}, store.lean, res);
    renderLeanPane();
  }).catch(apiFail);
}

function leanShare() {
  leanSave().then(function () {
    apiPost('/api/lean/share', {}).then(function (res) {
      store.lean = Object.assign({}, store.lean, res);
      renderLeanPane();
    }).catch(function (err) { loadLeanSettings(); apiFail(err); });
  }).catch(apiFail);
}

function leanStopShare() {
  apiPost('/api/lean/stop-share', {}).then(function (res) {
    store.lean = Object.assign({}, store.lean, res);
    renderLeanPane();
  }).catch(apiFail);
}

function leanCopyToken() {
  apiGet('/api/lean/token').then(function (res) {
    var token = (res && res.token) || '';
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(token).then(function () {
        toast('Access token copied');
      }, function () { toast('The token file is server-token in the server folder'); });
    } else {
      toast('The token file is server-token in the server folder');
    }
  }).catch(apiFail);
}

function leanOpenLogs() {
  apiPost('/api/lean/open-logs', {}).then(function () {
    toast('Opened the server folder');
  }).catch(apiFail);
}

function wireLeanPane() {
  if (!sheetEl) return;
  ['leanStorage', 'leanPath', 'leanNgrok'].forEach(function (id) {
    var input = $('#' + id, sheetEl);
    if (input) input.addEventListener('change', leanSave);
  });
  var pairs = { leanStart: leanStart, leanStop: leanStop, leanOpenLogs: leanOpenLogs,
                leanShare: leanShare, leanStopShare: leanStopShare,
                leanCopyToken: leanCopyToken };
  Object.keys(pairs).forEach(function (id) {
    var button = $('#' + id, sheetEl);
    if (button) button.addEventListener('click', pairs[id]);
  });
  renderLeanPane();
}

function antigravityAccessMapping(value) {
  if (value === 'unrestricted') return { antigravityPermission: 'auto', antigravityMode: 'accept-edits' };
  if (value === 'plan') return { antigravityPermission: 'review', antigravityMode: 'plan' };
  if (value === 'auto') return { antigravityPermission: 'review', antigravityMode: 'auto' };
  return { antigravityPermission: 'review', antigravityMode: 'accept-edits' };
}

function saveSettings() {
  var settings = draftSettings();
  var body = {};
  $all('[data-key]', sheetEl).forEach(function (input) {
    var key = input.getAttribute('data-key');
    var value = input.value.trim();
    if (key === 'antigravityAccess') {
      var mapped = antigravityAccessMapping(value);
      var changed = Object.keys(mapped).some(function (k) {
        return String(settings[k] != null ? settings[k] : '') !== mapped[k];
      });
      if (changed) {
        Object.keys(mapped).forEach(function (k) { body[k] = mapped[k]; settings[k] = mapped[k]; });
      }
      return;
    }
    if (String(settings[key] != null ? settings[key] : '') !== value) {
      body[key] = value;
      settings[key] = value;
    }
  });
  var requests = [];
  if (Object.keys(body).length > 0) {
    requests.push(apiPost('/api/settings', body).then(function () {
      store.settings = Object.assign({}, store.settings, body);
    }));
  }
  var editor = sheetEl && $('#policyEditor', sheetEl);
  if (editor && editor.value !== policyDraft) {
    requests.push(apiPost('/api/policy', { text: editor.value }).then(function () {
      policyDraft = editor.value;
    }));
  }
  requests.push(apiPost('/api/mcp', { enabled: mcpDraft.enabled }));
  if (!requests.length) { toast('Settings saved'); return Promise.resolve(); }
  return Promise.all(requests).then(function () {
    toast('Settings saved');
    renderHeader();
    return null;
  }).catch(function (err) { apiFail(err); throw err; });
}

function renderServerBox() {
  if (!sheetEl) return;
  var box = $('#serverBox', sheetEl);
  if (!box) return;
  var server = store.server || {};
  var protocol = location.protocol === 'https:' ? 'https://' : 'http://';
  var url = protocol + (location.host || 'localhost');
  var health = store.health || {};
  box.innerHTML =
    '<div class="server-box">' +
      '<div class="server-line"><span class="k">Host</span><span class="v">' + esc(location.host || 'localhost') + '</span></div>' +
      '<div class="server-line"><span class="k">Port</span><span class="v">' + esc(server.port != null ? server.port : '') + '</span></div>' +
      '<div class="server-line"><span class="k">Access mode</span><span class="v">' + esc(server.access || 'local') + '</span></div>' +
      '<div class="server-line"><span class="k">Platform</span><span class="v">' + esc(server.platform || '') + '</span></div>' +
      '<div class="server-line"><span class="k">LAN / Tailscale</span><span class="v server-url">' + esc(url) + '</span></div>' +
      '<div class="server-line"><span class="k">Engine</span><span class="v">' + esc(health.version ? 'v' + health.version + ' · up ' + Math.round(health.uptimeSeconds || 0) + 's' : '—') + '</span></div>' +
      (server.editors ? '<div class="server-line"><span class="k">Note</span><span class="v">' + esc(server.editors) + '</span></div>' : '') +
    '</div>';
}

// ===== Add project sheet ====================================================

function openAddSheet() {
  if (sheetEl) return;
  closeMenu();
  var backdrop = document.createElement('div');
  backdrop.className = 'sheet-backdrop';
  backdrop.innerHTML =
    '<div class="sheet sheet-add" role="dialog" aria-label="Add a project">' +
      '<div class="sheet-head"><span class="sheet-title" style="font-size:24px">Make room for a new idea.</span></div>' +
      '<div class="sheet-body" style="padding-top:12px">' +
        '<p class="section-note" style="margin-top:0">Add a GitHub repository or connect an existing project on the workspace drive.</p>' +
        '<div class="field"><label for="addRepoField">Repository or local folder</label>' +
          '<input id="addRepoField" placeholder="https://github.com/owner/repository" autocomplete="off" spellcheck="false"></div>' +
        '<div class="field"><label for="addInstructionsField">Research instructions (auto-research.md)</label>' +
          '<textarea id="addInstructionsField" rows="6" placeholder="Optional. Paste the project\u2019s research goal, constraints, budgets and expected behavior here; it is saved as the checkout\u2019s auto-research.md. Repository projects are cloned now when instructions are provided."></textarea></div>' +
      '</div>' +
      '<div class="sheet-foot"><span class="spacer"></span>' +
        '<button class="sheet-secondary" id="addCancel">Cancel</button>' +
        '<button class="sheet-primary" id="addConfirm" disabled>Add repository</button>' +
      '</div>' +
    '</div>';
  $('#sheetHost').appendChild(backdrop);
  sheetEl = backdrop;
  backdrop.addEventListener('mousedown', function (e) {
    if (e.target === backdrop) closeSheet();
  });
  var field = $('#addRepoField', backdrop);
  var guide = $('#addInstructionsField', backdrop);
  var confirm = $('#addConfirm', backdrop);
  field.addEventListener('input', function () { confirm.disabled = field.value.trim().length === 0; });
  field.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && field.value.trim()) { e.preventDefault(); submitAdd(); }
  });
  $('#addCancel', backdrop).addEventListener('click', closeSheet);
  confirm.addEventListener('click', submitAdd);
  setTimeout(function () { field.focus(); }, 60);

  function submitAdd() {
    confirm.disabled = true;
    addProjectFromInput(field.value, guide.value).then(function (res) {
      closeSheet();
      toast(res && res.seeded ? 'Project added · instructions saved' : 'Project added');
    }).catch(function (err) {
      confirm.disabled = false;
      apiFail(err);
    });
  }
}

// ===== Instructions & project editing ======================================

function openInstructionsSheet(projectID) {
  if (sheetEl) return;
  closeMenu();
  var project = store.projects.find(function (x) { return x.id === (projectID || store.selectedProjectID); });
  if (!project) { toast('Select a project first', true); return; }
  var backdrop = document.createElement('div');
  backdrop.className = 'sheet-backdrop';
  backdrop.innerHTML =
    '<div class="sheet sheet-instructions" role="dialog" aria-label="Edit instructions">' +
      '<div class="sheet-head"><span class="sheet-title">Edit research instructions</span><span class="spacer"></span>' +
        '<button class="sheet-done" id="instrDone">Done</button></div>' +
      '<div class="sheet-body">' +
        '<p class="section-note" id="instrPath">Loading…</p>' +
        '<textarea id="instrEditor" class="mono-editor" rows="16" spellcheck="false" aria-label="auto-research.md"></textarea>' +
        '<p class="section-note">The guide is created with a template when it does not exist yet and is never overwritten by a run, so your edits survive every run and every new chat.</p>' +
      '</div>' +
      '<div class="sheet-foot"><span class="sheet-note" id="instrNote"></span><span class="spacer"></span>' +
        '<button class="sheet-secondary" id="instrCancel">Cancel</button>' +
        '<button class="sheet-primary" id="instrSave">Save</button>' +
      '</div>' +
    '</div>';
  $('#sheetHost').appendChild(backdrop);
  sheetEl = backdrop;
  backdrop.addEventListener('mousedown', function (e) {
    if (e.target === backdrop) closeSheet();
  });
  $('#instrDone', backdrop).addEventListener('click', function () {
    saveInstructions(project.id).then(function () { closeSheet(); }).catch(function () { /* keep open */ });
  });
  $('#instrCancel', backdrop).addEventListener('click', closeSheet);
  $('#instrSave', backdrop).addEventListener('click', function () {
    saveInstructions(project.id);
  });
  apiGet('/api/instructions?projectID=' + encodeURIComponent(project.id)).then(function (res) {
    $('#instrPath', backdrop).textContent = res.path || '';
    $('#instrEditor', backdrop).value = res.text || '';
  }).catch(function (err) {
    closeSheet();
    apiFail(err);
  });
}

function saveInstructions(projectID) {
  var editor = sheetEl && $('#instrEditor', sheetEl);
  if (!editor) return Promise.resolve();
  $('#instrNote', sheetEl).textContent = 'Saving…';
  return apiPost('/api/instructions', { projectID: projectID, text: editor.value }).then(function () {
    $('#instrNote', sheetEl).textContent = '';
    toast('Instructions saved');
    setupSig = '';
    render();
    return null;
  }).catch(function (err) {
    var note = $('#instrNote', sheetEl);
    if (note) note.textContent = (err && err.message) || 'Save failed';
    throw err;
  });
}

function openProjectEditSheet(projectID) {
  if (sheetEl) return;
  closeMenu();
  var project = store.projects.find(function (x) { return x.id === projectID; });
  if (!project) return;
  var backdrop = document.createElement('div');
  backdrop.className = 'sheet-backdrop';
  backdrop.innerHTML =
    '<div class="sheet sheet-add" role="dialog" aria-label="Edit project details">' +
      '<div class="sheet-head"><span class="sheet-title" style="font-size:24px">Edit project details</span></div>' +
      '<div class="sheet-body" style="padding-top:12px">' +
        '<div class="field"><label for="editName">Name</label>' +
          '<input id="editName" value="' + attr(projectName(project)) + '"></div>' +
        '<div class="field"><label for="editSummary">Summary</label>' +
          '<textarea id="editSummary" rows="3">' + esc(project.summary || '') + '</textarea></div>' +
        '<div class="field"><label for="editCategory">Category</label>' +
          '<input id="editCategory" value="' + attr(project.category || 'Research') + '"></div>' +
        (project.builtinID ? '' :
          '<div class="field"><label for="editRepo">Repository (owner/name)</label>' +
            '<input id="editRepo" value="' + attr(project.repository || '') + '"' +
            (project.localPath ? ' disabled title="This project uses a local folder"' : '') + '></div>' +
          (project.localPath ?
            '<div class="field"><label for="editLocal">Local folder</label>' +
              '<input id="editLocal" value="' + attr(project.localPath || '') + '"></div>' : '')) +
      '</div>' +
      '<div class="sheet-foot"><span class="sheet-note">Chats keep their history.</span><span class="spacer"></span>' +
        '<button class="sheet-secondary" id="editCancel">Cancel</button>' +
        '<button class="sheet-primary" id="editSave">Save</button>' +
      '</div>' +
    '</div>';
  $('#sheetHost').appendChild(backdrop);
  sheetEl = backdrop;
  backdrop.addEventListener('mousedown', function (e) {
    if (e.target === backdrop) closeSheet();
  });
  $('#editCancel', backdrop).addEventListener('click', closeSheet);
  $('#editSave', backdrop).addEventListener('click', function () {
    var body = { projectID: project.id,
                 name: $('#editName', backdrop).value.trim(),
                 summary: $('#editSummary', backdrop).value.trim(),
                 category: $('#editCategory', backdrop).value.trim() };
    var repoField = $('#editRepo', backdrop);
    var localField = $('#editLocal', backdrop);
    if (repoField && !repoField.disabled && repoField.value.trim()) body.repository = repoField.value.trim();
    if (localField && localField.value.trim()) body.localPath = localField.value.trim();
    $('#editSave', backdrop).disabled = true;
    apiPost('/api/projects/edit', body).then(function (res) {
      closeSheet();
      toast('Project updated');
      if (res && res.project) {
        var found = store.projects.find(function (x) { return x.id === body.projectID; });
        if (found) Object.assign(found, res.project);
        render();
      } else {
        refreshCatalog();
      }
    }).catch(function (err) {
      $('#editSave', backdrop).disabled = false;
      apiFail(err);
    });
  });
}

function openRemoveProjectSheet(projectID) {
  if (sheetEl) return;
  closeMenu();
  var project = store.projects.find(function (x) { return x.id === projectID; });
  if (!project) return;
  var backdrop = document.createElement('div');
  backdrop.className = 'sheet-backdrop';
  backdrop.innerHTML =
    '<div class="sheet sheet-add" role="dialog" aria-label="Remove project">' +
      '<div class="sheet-head"><span class="sheet-title" style="font-size:24px">Remove ' + esc(projectName(project)) + '?</span></div>' +
      '<div class="sheet-body" style="padding-top:12px">' +
        '<p class="section-note" style="margin-top:0">The saved project entry is removed from this workspace. Files on disk and your CLI data are not touched. Close this project’s open chats first; their closed history is preserved.</p>' +
      '</div>' +
      '<div class="sheet-foot"><span class="spacer"></span>' +
        '<button class="sheet-secondary" id="rmCancel">Cancel</button>' +
        '<button class="sheet-primary" id="rmConfirm">Remove project</button>' +
      '</div>' +
    '</div>';
  $('#sheetHost').appendChild(backdrop);
  sheetEl = backdrop;
  backdrop.addEventListener('mousedown', function (e) {
    if (e.target === backdrop) closeSheet();
  });
  $('#rmCancel', backdrop).addEventListener('click', closeSheet);
  $('#rmConfirm', backdrop).addEventListener('click', function () {
    apiPost('/api/projects/remove', { projectID: project.id }).then(function () {
      closeSheet();
      toast('Project removed');
      if (store.selectedProjectID === project.id) {
        store.selectedProjectID = null;
        store.selectedSessionID = null;
      }
      store.projects = store.projects.filter(function (x) { return x.id !== project.id; });
      render();
    }).catch(function (err) {
      closeSheet();
      apiFail(err);
    });
  });
}

// ===== Events & boot ========================================================

function wireEvents() {
  var search = $('#searchInput');
  search.addEventListener('input', function () {
    store.search = search.value;
    $('#searchClear').hidden = search.value.length === 0;
    renderSidebar();
  });
  $('#searchClear').addEventListener('click', function () {
    search.value = '';
    store.search = '';
    $('#searchClear').hidden = true;
    search.focus();
    renderSidebar();
  });
  $('#sideNewChat').addEventListener('click', function () {
    createChat(store.selectedProjectID, currentProvider());
  });
  $('#refreshBtn').addEventListener('click', refreshCatalog);
  $('#addBtn').addEventListener('click', openAddSheet);
  $('#settingsBtn').addEventListener('click', openSettingsSheet);

  $('#sideList').addEventListener('click', function (e) {
    var closed = e.target.closest('#closedMenuBtn');
    if (closed) { openClosedChatsMenu(closed); return; }
    var play = e.target.closest('[data-play]');
    if (play) {
      var p = store.projects.find(function (x) { return x.id === play.getAttribute('data-play'); });
      if (p) playProject(p);
      return;
    }
    var sel = e.target.closest('[data-select]');
    if (sel) { selectProject(sel.getAttribute('data-select')); return; }
    var ses = e.target.closest('[data-session]');
    if (ses) { selectSession(ses.getAttribute('data-session')); return; }
    var close = e.target.closest('[data-close]');
    if (close) { closeSession(close.getAttribute('data-close')); return; }
  });

  $('#header').addEventListener('click', headerClick);

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && sheetEl) { e.preventDefault(); closeSheet(); }
    if ((e.metaKey || e.ctrlKey) && (e.key === 'n' || e.key === 'N')) {
      e.preventDefault();
      createChat(store.selectedProjectID, currentProvider());
    }
  });

  window.addEventListener('resize', function () {
    closeMenu();
    var body = $('#mainBody');
    if (body && body.__mode === 'dashboard') { dashSig = ''; renderMain(); }
  });
}

function render() {
  if (!store.loaded) return;
  renderSidebar();
  renderHeader();
  renderMain();
  updateSettingsSheetAuth();
}

function boot() {
  injectStaticIcons();
  var params = new URLSearchParams(window.location.search);
  var fromURL = params.get('token');
  if (fromURL) {
    store.token = fromURL;
    try { localStorage.setItem(TOKEN_KEY, fromURL); } catch (e) { /* ignore */ }
    try { window.history.replaceState(null, '', window.location.pathname); } catch (e) { /* ignore */ }
  } else {
    try { store.token = localStorage.getItem(TOKEN_KEY) || ''; } catch (e) { store.token = ''; }
  }
  wireEvents();
  pollLoop();
}

boot();

})();
