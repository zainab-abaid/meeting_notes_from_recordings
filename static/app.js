const state = {
  recordings: [],
  selectedId: null,
  analysis: null,
  jobTimer: null,
  jobId: null,
  search: "",
  filter: "all",
  groupOpen: {},
};

const els = {
  list: document.getElementById("recording-list"),
  search: document.getElementById("meeting-search"),
  filters: document.getElementById("meeting-filters"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  empty: document.getElementById("empty-state"),
  workspace: document.getElementById("workspace"),
  title: document.getElementById("meeting-title"),
  date: document.getElementById("meeting-date"),
  meta: document.getElementById("meeting-meta"),
  analyzeBtn: document.getElementById("analyze-btn"),
  meetBtn: document.getElementById("meet-btn"),
  meetRemove: document.getElementById("meet-remove"),
  stopBtn: document.getElementById("stop-btn"),
  meetInput: document.getElementById("meet-input"),
  meetStatus: document.getElementById("meet-status"),
  progress: document.getElementById("progress"),
  progressFill: document.getElementById("progress-fill"),
  progressLabel: document.getElementById("progress-label"),
  player: document.getElementById("player"),
  playerMissing: document.getElementById("player-missing"),
  nowPlaying: document.getElementById("now-playing"),
  tabs: document.getElementById("tabs"),
  status: document.getElementById("status-line"),
};

const speakerColors = ["#c45c26", "#3f6f58", "#3d5a80", "#8a4f2a", "#6b4c9a", "#8a6d2b"];

els.dropzone.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", () => {
  if (els.fileInput.files[0]) uploadFile(els.fileInput.files[0]);
});

["dragenter", "dragover"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove("dragover");
  });
});
els.dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) uploadFile(file);
});

document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("drop", (event) => {
  if (event.target.closest(".dropzone")) return;
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  if (file) uploadFile(file);
});

els.tabs.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-tab]");
  if (!button) return;
  setTab(button.dataset.tab);
});

els.analyzeBtn.addEventListener("click", () => {
  startAnalysis(state.selectedId, {
    force: false,
    reanalyze: Boolean(state.analysis),
  });
});
els.stopBtn.addEventListener("click", () => stopAnalysis());

els.search.addEventListener("input", () => {
  state.search = els.search.value;
  renderList();
});
els.filters.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-filter]");
  if (!button) return;
  state.filter = button.dataset.filter;
  for (const item of els.filters.querySelectorAll("button")) {
    item.classList.toggle("active", item === button);
  }
  renderList();
});
els.list.addEventListener("keydown", (event) => {
  if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
  const ids = [];
  for (const group of groupedRecordings(visibleRecordings())) {
    if (!isGroupOpen(group.label, group.items)) continue;
    for (const recording of group.items) ids.push(recording.id);
  }
  if (!ids.length) return;
  event.preventDefault();
  const index = Math.max(0, ids.indexOf(state.selectedId));
  const next = event.key === "ArrowDown" ? index + 1 : index - 1;
  if (next >= 0 && next < ids.length) selectRecording(ids[next], { scrollList: true });
});

els.meetBtn.addEventListener("click", () => {
  if (!state.selectedId) {
    showStatus("Select a recording first.", true);
    return;
  }
  els.meetInput.click();
});
els.meetInput.addEventListener("change", () => {
  if (els.meetInput.files[0]) uploadMeetTranscript(els.meetInput.files[0]);
});
els.meetRemove.addEventListener("click", () => {
  if (!state.selectedId) return;
  removeMeetTranscript();
});

document.body.addEventListener("click", (event) => {
  const cite = event.target.closest("[data-start]");
  if (!cite) return;
  const seconds = Number(cite.dataset.start);
  if (Number.isFinite(seconds)) seekTo(seconds);
});

els.player.addEventListener("timeupdate", () => {
  highlightTranscript(els.player.currentTime);
});

async function loadRecordings(selectId) {
  const data = await fetchJSON("/api/recordings");
  state.recordings = data.recordings || [];
  const firstLoad = !state.selectedId;
  renderList();
  const inFlight = state.recordings.find(
    (recording) => recording.job && ["queued", "running", "cancelling"].includes(recording.job.status)
  );
  const target = selectId || state.selectedId || inFlight?.id || state.recordings[0]?.id;
  if (target) await selectRecording(target, { scrollList: Boolean(selectId) || firstLoad });
}

async function selectRecording(id, { scrollList = false } = {}) {
  const changed = state.selectedId !== id;
  state.selectedId = id;
  renderList({ scrollList: scrollList || changed });
  const data = await fetchJSON(`/api/recordings/${encodeURIComponent(id)}`);
  state.analysis = data.analysis;
  renderWorkspace(data.recording, data.analysis);
  if (data.recording?.job && ["queued", "running", "cancelling"].includes(data.recording.job.status)) {
    watchJob(data.recording.job.id);
  }
}

function renderList({ scrollList = false } = {}) {
  els.list.innerHTML = "";
  if (!state.recordings.length) {
    els.list.innerHTML = `<li class="muted" style="padding:0.4rem 0.2rem">Nothing here yet.</li>`;
    return;
  }
  const matches = visibleRecordings();
  if (!matches.length) {
    els.list.innerHTML = `<li class="muted" style="padding:0.4rem 0.2rem">No meetings match that search.</li>`;
    return;
  }
  for (const group of groupedRecordings(matches)) {
    const open = isGroupOpen(group.label, group.items);
    const heading = document.createElement("li");
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "group-toggle";
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    toggle.innerHTML = `<span>${open ? "▾" : "▸"} ${escapeHtml(group.label)}</span><span class="group-count">${group.items.length}</span>`;
    toggle.addEventListener("click", () => {
      if (open && group.items.some((recording) => recording.id === state.selectedId)) return;
      state.groupOpen[group.label] = !open;
      renderList();
    });
    heading.appendChild(toggle);
    els.list.appendChild(heading);
    if (!open) continue;
    for (const recording of group.items) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = recording.id === state.selectedId ? "active" : "";
      button.title = displayTitle(recording);
      const job = recording.job;
      let pill = recording.analyzed
        ? `<span class="pill ready">Analyzed</span>`
        : `<span class="pill wait">To review</span>`;
      if (job && ["queued", "running", "cancelling"].includes(job.status)) {
        pill = `<span class="pill run">Analyzing</span>`;
      }
      button.innerHTML = `
        <span class="title">${escapeHtml(displayTitle(recording))}</span>
        <span class="sub"><span>${escapeHtml(formatListDate(recording))}</span>${pill}</span>
      `;
      button.addEventListener("click", () => selectRecording(recording.id));
      item.appendChild(button);
      els.list.appendChild(item);
    }
  }
  if (scrollList) {
    els.list.querySelector("button.active")?.scrollIntoView({ block: "nearest" });
  }
}

function groupedRecordings(matches) {
  const groups = [];
  for (const recording of matches) {
    const label = meetingGroup(recording);
    const last = groups[groups.length - 1];
    if (!last || last.label !== label) {
      groups.push({ label, items: [recording] });
    } else {
      last.items.push(recording);
    }
  }
  return groups;
}

function isGroupOpen(label, items) {
  if (state.search.trim()) return true;
  if (items.some((recording) => recording.id === state.selectedId)) return true;
  if (Object.prototype.hasOwnProperty.call(state.groupOpen, label)) {
    return Boolean(state.groupOpen[label]);
  }
  return ["In progress", "Today", "Yesterday", "This week"].includes(label);
}

function visibleRecordings() {
  const query = state.search.trim().toLowerCase();
  const filtered = state.recordings.filter((recording) => {
    if (state.filter === "ready" && !recording.analyzed) return false;
    if (state.filter === "wait" && recording.analyzed) return false;
    if (!query) return true;
    const haystack = [
      displayTitle(recording),
      recording.title,
      recording.filename,
      recording.date,
      ...(recording.participants || []),
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });
  const inProgress = [];
  const rest = [];
  for (const recording of filtered) {
    const job = recording.job;
    if (job && ["queued", "running", "cancelling"].includes(job.status)) {
      inProgress.push(recording);
    } else {
      rest.push(recording);
    }
  }
  return [...inProgress, ...rest];
}

function displayTitle(recording) {
  let title = String(recording.title || recording.filename || "").trim();
  title = title.replace(/\s*-\s*Recording$/i, "");
  title = title.replace(/\s*-\s*\d{4}[_-]\d{2}[_-]\d{2}.*$/i, "");
  return title.trim() || recording.filename || "Meeting";
}

function meetingDate(recording) {
  if (recording.date) {
    const parsed = new Date(recording.date);
    if (!Number.isNaN(parsed.getTime())) return parsed;
  }
  const fromName = String(recording.filename || recording.id || "").match(
    /(\d{4})[_-](\d{2})[_-](\d{2})/
  );
  if (fromName) {
    const parsed = new Date(Number(fromName[1]), Number(fromName[2]) - 1, Number(fromName[3]));
    if (!Number.isNaN(parsed.getTime())) return parsed;
  }
  if (recording.when) return new Date(Number(recording.when) * 1000);
  return new Date();
}

function formatListDate(recording) {
  const date = meetingDate(recording);
  return date.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

function meetingGroup(recording) {
  const job = recording.job;
  if (job && ["queued", "running", "cancelling"].includes(job.status)) {
    return "In progress";
  }
  const date = meetingDate(recording);
  const now = new Date();
  const startOfDay = (value) => new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
  const day = startOfDay(date);
  const today = startOfDay(now);
  const yesterday = today - 86400000;
  const weekAgo = today - 6 * 86400000;
  if (day === today) return "Today";
  if (day === yesterday) return "Yesterday";
  if (day >= weekAgo) return "This week";
  if (date.getFullYear() === now.getFullYear() && date.getMonth() === now.getMonth()) {
    return "Earlier this month";
  }
  return date.toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

function renderWorkspace(recording, analysis) {
  els.empty.classList.add("hidden");
  els.workspace.classList.remove("hidden");
  els.title.textContent = analysis?.title || recording.title || recording.filename;
  els.date.textContent = analysis?.date || recording.date || "Meeting";
  const people = (analysis?.participants || recording.participants || []).join(", ");
  els.meta.textContent = people ? people : recording.has_media ? "Speakers not identified yet" : "Notes only — video file missing";
  if (recording.has_meet_transcript) {
    els.meetStatus.textContent =
      "Meet PDF attached — used to name speakers and as extra context. The audio transcript still wins if they disagree.";
    els.meetBtn.textContent = "Replace Meet PDF";
    els.meetRemove.classList.remove("hidden");
  } else {
    els.meetStatus.textContent =
      "Optional: attach Gemini’s Google Meet transcript as a PDF to name speakers. Without it, speakers stay Speaker A, B, C.";
    els.meetBtn.textContent = "Attach Meet PDF";
    els.meetRemove.classList.add("hidden");
  }

  const job = recording.job;
  const running = job && ["queued", "running", "cancelling"].includes(job.status);
  const cancelled = job && job.status === "cancelled";
  els.analyzeBtn.disabled = running || !recording.has_media;
  els.stopBtn.classList.toggle("hidden", !running);
  els.stopBtn.disabled = Boolean(job && job.status === "cancelling");
  els.stopBtn.textContent = job && job.status === "cancelling" ? "Stopping…" : "Stop";
  if (!recording.has_media) {
    els.analyzeBtn.textContent = "Add video to analyze";
  } else if (running) {
    els.analyzeBtn.textContent = job.status === "cancelling" ? "Stopping…" : "Analyzing…";
  } else if (cancelled && !analysis) {
    els.analyzeBtn.textContent = "Restart analysis";
  } else if (analysis) {
    els.analyzeBtn.textContent = analysis.legacy ? "Re-analyze for timestamps" : "Re-analyze";
  } else {
    els.analyzeBtn.textContent = "Analyze";
  }

  if (running) {
    showProgress(job.progress || 0.05, job.message || "Working…");
    hideStatus();
  } else {
    hideProgress();
  }

  if (recording.has_media) {
    els.player.classList.remove("hidden");
    els.playerMissing.classList.add("hidden");
    const mediaUrl = `/api/recordings/${encodeURIComponent(recording.id)}/media`;
    if (els.player.dataset.src !== mediaUrl) {
      els.player.src = mediaUrl;
      els.player.dataset.src = mediaUrl;
    }
  } else {
    els.player.removeAttribute("src");
    els.player.load();
    els.player.classList.add("hidden");
    els.playerMissing.classList.remove("hidden");
  }

  renderAnalysis(analysis, { running, cancelled });
}

function renderAnalysis(analysis, { running = false, cancelled = false } = {}) {
  const summary = document.getElementById("tab-summary");
  const notes = document.getElementById("tab-notes");
  const actions = document.getElementById("tab-actions");
  const decisions = document.getElementById("tab-decisions");
  const transcript = document.getElementById("tab-transcript");

  if (!analysis) {
    const ctaLabel = running ? "Analyzing…" : cancelled ? "Restart analysis" : "Analyze this recording";
    const empty = `
      <div class="cta">
        <h3>This recording has not been analyzed yet</h3>
        <p class="muted">Transcription usually takes a few minutes. Progress will show above the tabs.</p>
        <p class="muted">Optional: attach a Google Meet transcript PDF first if you want real speaker names. Without it, people stay Speaker A, B, C — names are not guessed.</p>
        <button class="ghost" id="meet-cta" type="button">Attach Meet PDF</button>
        <button class="primary" id="analyze-cta" type="button" ${running ? "disabled" : ""}>${ctaLabel}</button>
      </div>`;
    summary.innerHTML = empty;
    notes.innerHTML = `<p class="muted">No notes yet. Analyze this recording first.</p>`;
    actions.innerHTML = `<p class="muted">No action items yet. Analyze this recording first.</p>`;
    decisions.innerHTML = `<p class="muted">No decisions yet. Analyze this recording first.</p>`;
    transcript.innerHTML = `<p class="muted">No transcript yet.</p>`;
    document.getElementById("meet-cta")?.addEventListener("click", () => {
      if (!state.selectedId) {
        showStatus("Select a recording first.", true);
        return;
      }
      els.meetInput.click();
    });
    document.getElementById("analyze-cta")?.addEventListener("click", () => {
      if (running) return;
      startAnalysis(state.selectedId, { force: false });
    });
    return;
  }

  const legacy = analysis.legacy
    ? `<p class="banner">These notes were generated before timestamped citations. Re-analyze to make each point jump to the exact moment in the recording. The transcript may also still contain Urdu in another script until you re-analyze.</p>`
    : "";

  summary.innerHTML = `${legacy}<h3>Overview</h3>${renderCitedText(analysis.overview)}${renderQuestions(analysis.open_questions)}`;
  notes.innerHTML = renderNotes(analysis.discussion_notes);
  actions.innerHTML = renderActions(analysis.action_items);
  decisions.innerHTML = renderDecisions(analysis.decisions);
  transcript.innerHTML = renderTranscript(analysis.transcript || []);
}

function renderCitedText(items) {
  if (!items || !items.length) return `<p class="muted">Nothing here yet.</p>`;
  return `<p>${items.map((item) => `${escapeHtml(item.text || "")}${citeButton(item.start)}`).join(" ")}</p>`;
}

function renderNotes(blocks) {
  if (!blocks || !blocks.length) return `<p class="muted">No discussion notes.</p>`;
  return blocks
    .map((block) => {
      const items = (block.items || [])
        .map((item) => `<li>${escapeHtml(item.text || "")}${citeButton(item.start)}</li>`)
        .join("");
      return `<h3>${escapeHtml(block.topic || "Discussion")}</h3><ul>${items || "<li class='muted'>No points captured.</li>"}</ul>`;
    })
    .join("");
}

function renderActions(items) {
  if (!items || !items.length) return `<p class="muted">No action items were assigned.</p>`;
  return `<ul>${items
    .map((item) => {
      const owner = item.owner || "Unassigned";
      const due = item.due ? ` <span class="muted">Due ${escapeHtml(item.due)}</span>` : "";
      return `<li><strong>${escapeHtml(owner)}:</strong> ${escapeHtml(item.task || "")}${due}${citeButton(item.start)}</li>`;
    })
    .join("")}</ul>`;
}

function renderDecisions(items) {
  if (!items || !items.length) return `<p class="muted">No decisions were recorded.</p>`;
  return `<ul>${items
    .map((item) => {
      const actor = item.actor ? `<strong>${escapeHtml(item.actor)}:</strong> ` : "";
      return `<li>${actor}${escapeHtml(item.text || "")}${citeButton(item.start)}</li>`;
    })
    .join("")}</ul>`;
}

function renderQuestions(items) {
  if (!items || !items.length) return "";
  return `<h3>Open questions</h3><ul>${items
    .map((item) => `<li>${escapeHtml(item.text || "")}${citeButton(item.start)}</li>`)
    .join("")}</ul>`;
}

function renderTranscript(segments) {
  if (!segments.length) return `<p class="muted">No transcript yet.</p>`;
  const speakers = [...new Set(segments.map((segment) => segment.speaker || "Speaker"))];
  return `<div class="transcript">${segments
    .map((segment, index) => {
      const speaker = segment.speaker || "Speaker";
      const color = speakerColors[speakers.indexOf(speaker) % speakerColors.length];
      return `<div class="utterance" data-index="${index}" data-start="${segment.start || 0}">
        <button class="when" data-start="${segment.start || 0}" type="button">${formatTime(segment.start || 0)}</button>
        <div class="who" style="color:${color}">${escapeHtml(speaker)}</div>
        <p class="said">${escapeHtml(segment.text || "")}</p>
      </div>`;
    })
    .join("")}</div>`;
}

function citeButton(start) {
  if (start === null || start === undefined || start === "") return "";
  const seconds = Number(start);
  if (!Number.isFinite(seconds)) return "";
  return `<button class="cite" type="button" data-start="${seconds}">${formatTime(seconds)}</button>`;
}

function setTab(name) {
  for (const button of els.tabs.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.tab === name);
  }
  for (const panel of document.querySelectorAll(".tab-panel")) {
    panel.classList.toggle("hidden", panel.id !== `tab-${name}`);
  }
}

function seekTo(seconds) {
  if (els.player.classList.contains("hidden")) return;
  els.player.currentTime = Math.max(0, seconds);
  els.player.play().catch(() => {});
  els.nowPlaying.textContent = `Playing from ${formatTime(seconds)}`;
  const onTranscript = !document.getElementById("tab-transcript").classList.contains("hidden");
  highlightTranscript(seconds, onTranscript);
}

function highlightTranscript(current, scroll) {
  const rows = document.querySelectorAll(".utterance");
  let active = null;
  for (const row of rows) {
    const start = Number(row.dataset.start);
    if (start <= current + 0.25) active = row;
  }
  for (const row of rows) row.classList.toggle("active", row === active);
  if (scroll && active) active.scrollIntoView({ block: "center", behavior: "smooth" });
}

async function uploadFile(file) {
  els.dropzone.querySelector("strong").textContent = "Uploading…";
  const body = new FormData();
  body.append("file", file);
  try {
    const data = await fetchJSON("/api/upload", { method: "POST", body });
    const id = data.recording.id;
    await loadRecordings(id);
    if (!data.recording.analyzed) await startAnalysis(id, { force: false });
  } catch (error) {
    alert(error.message);
    showStatus(error.message, true);
  } finally {
    els.dropzone.querySelector("strong").textContent = "Drop a recording";
    els.fileInput.value = "";
  }
}

async function uploadMeetTranscript(file) {
  if (!state.selectedId) return;
  els.meetBtn.textContent = "Attaching…";
  const body = new FormData();
  body.append("file", file);
  try {
    const data = await fetchJSON(
      `/api/recordings/${encodeURIComponent(state.selectedId)}/meet-transcript`,
      { method: "POST", body }
    );
    showStatus("Meet PDF attached. Re-analyze to apply speaker names and extra context.");
    const updated = data.recording;
    state.recordings = state.recordings.map((item) =>
      item.id === updated.id ? { ...item, ...updated } : item
    );
    renderList();
    await selectRecording(state.selectedId);
  } catch (error) {
    showStatus(error.message, true);
  } finally {
    els.meetInput.value = "";
  }
}

async function removeMeetTranscript() {
  try {
    const data = await fetchJSON(
      `/api/recordings/${encodeURIComponent(state.selectedId)}/meet-transcript`,
      { method: "DELETE" }
    );
    showStatus("Meet PDF removed. Re-analyze if you want Speaker A/B labels again.");
    const updated = data.recording;
    state.recordings = state.recordings.map((item) =>
      item.id === updated.id ? { ...item, ...updated } : item
    );
    renderList();
    await selectRecording(state.selectedId);
  } catch (error) {
    showStatus(error.message, true);
  }
}

async function startAnalysis(id, { force = false, reanalyze = false } = {}) {
  if (!id) {
    showStatus("Select a recording first.", true);
    return;
  }
  if (state.jobTimer && state.jobId) {
    return;
  }
  const cta = document.getElementById("analyze-cta");
  if (cta) {
    cta.disabled = true;
    cta.textContent = "Analyzing…";
  }
  showProgress(0.04, "Queuing analysis…");
  hideStatus();
  els.analyzeBtn.disabled = true;
  els.analyzeBtn.textContent = "Analyzing…";
  els.stopBtn.classList.remove("hidden");
  try {
    let data;
    try {
      data = await fetchJSON("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, force: Boolean(force), reanalyze: Boolean(reanalyze) }),
      });
    } catch (error) {
      const message = String(error.message || "");
      if (!message.includes("404") && !message.includes("Not Found")) throw error;
      data = await fetchJSON(`/api/recordings/${encodeURIComponent(id)}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: Boolean(force), reanalyze: Boolean(reanalyze) }),
      });
    }
    if (!data.job?.id) throw new Error("The server did not start an analysis job.");
    watchJob(data.job.id);
  } catch (error) {
    hideProgress();
    showStatus(error.message, true);
    await selectRecording(id);
  }
}

async function stopAnalysis() {
  const jobId = state.jobId;
  if (!jobId) return;
  els.stopBtn.disabled = true;
  els.stopBtn.textContent = "Stopping…";
  try {
    const data = await fetchJSON(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST",
    });
    showProgress(data.job?.progress || 0, data.job?.message || "Stopping…");
  } catch (error) {
    showStatus(error.message, true);
    els.stopBtn.disabled = false;
    els.stopBtn.textContent = "Stop";
  }
}

function watchJob(jobId) {
  if (state.jobTimer) clearInterval(state.jobTimer);
  state.jobId = jobId;
  const tick = async () => {
    try {
      const data = await fetchJSON(`/api/jobs/${jobId}`);
      const job = data.job;
      showProgress(job.progress || 0, job.message || "Working…");
      hideStatus();
      if (job.status === "done") {
        clearInterval(state.jobTimer);
        state.jobTimer = null;
        state.jobId = null;
        hideProgress();
        showStatus("Notes ready.");
        await loadRecordings(state.selectedId);
      } else if (job.status === "cancelled") {
        clearInterval(state.jobTimer);
        state.jobTimer = null;
        state.jobId = null;
        hideProgress();
        els.stopBtn.classList.add("hidden");
        els.stopBtn.disabled = false;
        els.stopBtn.textContent = "Stop";
        showStatus("Analysis stopped. Click Restart analysis to try again.");
        await loadRecordings(state.selectedId);
      } else if (job.status === "error") {
        clearInterval(state.jobTimer);
        state.jobTimer = null;
        state.jobId = null;
        hideProgress();
        showStatus(job.error || "Analysis failed", true);
        await loadRecordings(state.selectedId);
      }
    } catch (error) {
      clearInterval(state.jobTimer);
      state.jobTimer = null;
      state.jobId = null;
      hideProgress();
      showStatus(error.message, true);
    }
  };
  tick();
  state.jobTimer = setInterval(tick, 1200);
}

function hideStatus() {
  showStatus("");
}

function showStatus(message, isError = false) {
  if (!els.status) return;
  if (!message) {
    els.status.classList.add("hidden");
    els.status.textContent = "";
    return;
  }
  els.status.classList.remove("hidden");
  els.status.classList.toggle("error", Boolean(isError));
  els.status.textContent = message;
}

function showProgress(fraction, message) {
  els.progress.classList.remove("hidden");
  els.progressFill.style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  els.progressLabel.textContent = message;
}

function hideProgress() {
  els.progress.classList.add("hidden");
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatDetail(data.detail) || data.error || `Request failed (${response.status})`);
  }
  return data;
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours) return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  return `${minutes}:${String(secs).padStart(2, "0")}`;
}

function formatDetail(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  }
  return JSON.stringify(detail);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

loadRecordings();

setInterval(() => {
  if (!state.selectedId || state.jobTimer) return;
  fetchJSON(`/api/recordings/${encodeURIComponent(state.selectedId)}`)
    .then((data) => {
      const job = data.recording?.job;
      if (job && ["queued", "running", "cancelling"].includes(job.status)) {
        watchJob(job.id);
      }
    })
    .catch(() => {});
}, 2000);
