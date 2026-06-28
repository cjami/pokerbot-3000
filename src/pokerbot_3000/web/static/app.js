const apiStatus = document.querySelector("[data-api-status]");
const eventList = document.querySelector("[data-event-list]");
const gameControls = document.querySelectorAll("[data-game-control]");
const cameraForm = document.querySelector("[data-camera-form]");
const cameraDevice = document.querySelector("[data-camera-device]");
const cameraStatus = document.querySelector("[data-camera-status]");
const cameraPreview = document.querySelector("[data-public-camera-preview]");

const FRAME_INTERVAL_MS = 1500;
const FRAME_WIDTH = 1600;
const FRAME_MIME_TYPE = "image/png";

const spokenEventIds = new Set();
const speechInFlight = new Set();
const queuedSpeechEventIds = new Set();
const speechQueue = [];
let cameraStream = null;
let frameTimer = null;
let frameInFlight = false;
let speechQueueActive = false;
let currentSpeechAudio = null;
let shouldPlayInitialSnapshotSpeech = false;
let shouldSubmitBoardFrames = false;
let hasRenderedInitialSnapshot = false;

const rankLabels = {
  ace: "Ace",
  king: "King",
  queen: "Queen",
  jack: "Jack",
  "10": "10",
  "9": "9",
  "8": "8",
  "7": "7",
  "6": "6",
  "5": "5",
  "4": "4",
  "3": "3",
  "2": "2",
};

const suitLabels = {
  spades: "Spades",
  hearts: "Hearts",
  diamonds: "Diamonds",
  clubs: "Clubs",
};

function text(selector, value) {
  const element = document.querySelector(selector);
  if (element) {
    element.textContent = value;
  }
}

function cardLabel(card) {
  return `${rankLabels[card.rank] ?? card.rank} of ${suitLabels[card.suit] ?? card.suit}`;
}

function percent(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function renderCards(container, cards, emptyText) {
  if (!container) {
    return;
  }

  container.replaceChildren();
  if (!cards || cards.length === 0) {
    const empty = document.createElement("span");
    empty.className = "empty-state";
    empty.textContent = emptyText;
    container.append(empty);
    return;
  }

  for (const card of cards) {
    const item = document.createElement("span");
    item.className = "playing-card";
    item.textContent = cardLabel(card);
    container.append(item);
  }
}

function renderSeats(state) {
  const grid = document.querySelector("[data-seat-grid]");
  if (!grid) {
    return;
  }

  grid.replaceChildren();
  for (const [seat, player] of Object.entries(state.players)) {
    const article = document.createElement("article");
    article.className = Number(seat) === state.active_player_seat ? "seat-panel seat-panel-active" : "seat-panel";

    const header = document.createElement("div");
    header.className = "flex items-start justify-between gap-3";

    const identity = document.createElement("div");
    const seatLabel = document.createElement("p");
    seatLabel.className = "text-sm font-semibold text-neutral-600";
    seatLabel.textContent = `S${seat}`;
    const name = document.createElement("h3");
    name.className = "text-xl font-black";
    name.textContent = player.name;
    identity.append(seatLabel, name);
    header.append(identity);

    if (Number(seat) === state.dealer_seat) {
      const dealer = document.createElement("span");
      dealer.className = "dealer-button";
      dealer.textContent = "D";
      header.append(dealer);
    }

    const stats = document.createElement("dl");
    stats.className = "mt-4 grid gap-2 text-sm";
    stats.append(
      rowStat("Status", player.status),
      rowStat("Stack", String(player.stack)),
      rowStat("Committed", String(player.committed_this_street)),
    );
    article.append(header, stats);
    grid.append(article);
  }
}

function rowStat(label, value) {
  const row = document.createElement("div");
  row.className = "row-stat";
  const term = document.createElement("dt");
  term.textContent = label;
  const detail = document.createElement("dd");
  detail.textContent = value;
  row.append(term, detail);
  return row;
}

function renderRecognition(recognition) {
  const latest = recognition.latest_observation;
  text("[data-recognition-status]", recognition.status);
  text("[data-recognition-expected]", recognition.expected_card_count ?? "none");
  text("[data-recognition-stable]", `${recognition.stable_sample_count}/${recognition.required_stable_samples}`);
  text("[data-recognition-confidence]", percent(latest?.confidence ?? 0));
  text("[data-recognition-error]", recognition.last_error ?? "none");
  renderCards(document.querySelector("[data-detected-board]"), latest?.board_cards ?? [], "No detection yet");
}

function renderPrivateStates(privateStates) {
  const container = document.querySelector("[data-private-recognition]");
  if (!container) {
    return;
  }

  container.replaceChildren();
  for (const state of privateStates) {
    const row = document.createElement("div");
    row.className = "status-row";
    const copy = document.createElement("div");
    const name = document.createElement("p");
    name.className = "font-bold";
    name.textContent = state.agent_id.charAt(0).toUpperCase() + state.agent_id.slice(1);
    const detail = document.createElement("p");
    detail.className = "text-sm text-neutral-600";
    detail.textContent = `S${state.seat} - ${state.hole_cards.length}/2 cards`;
    copy.append(name, detail);
    const confidence = document.createElement("span");
    confidence.className = "status-pill";
    confidence.textContent = percent(state.confidence);
    row.append(copy, confidence);
    container.append(row);
  }
}

function renderClients(clients) {
  const container = document.querySelector("[data-clients]");
  if (!container) {
    return;
  }

  container.replaceChildren();
  for (const client of clients) {
    const row = document.createElement("div");
    row.className = "status-row";
    const copy = document.createElement("div");
    const name = document.createElement("p");
    name.className = "font-bold";
    name.textContent = client.client_id.replaceAll("_", " ");
    const detail = document.createElement("p");
    detail.className = "text-sm text-neutral-600";
    detail.textContent = client.status;
    copy.append(name, detail);
    const connection = document.createElement("span");
    connection.className = "status-pill";
    connection.textContent = client.connection;
    row.append(copy, connection);
    container.append(row);
  }
}

function renderActions(actions) {
  const container = document.querySelector("[data-legal-actions]");
  if (!container) {
    return;
  }

  container.replaceChildren();
  for (const action of actions) {
    const item = document.createElement("span");
    item.className = "action-chip";
    item.textContent = action;
    container.append(item);
  }
}

function renderUncertainties(uncertainties) {
  const list = document.querySelector("[data-uncertainties]");
  if (!list) {
    return;
  }

  list.replaceChildren();
  for (const uncertainty of uncertainties) {
    const item = document.createElement("li");
    item.textContent = uncertainty;
    list.append(item);
  }
}

function renderEvents(events) {
  if (!eventList) {
    return;
  }

  eventList.replaceChildren();
  for (const event of [...events].reverse()) {
    const item = document.createElement("li");
    const type = document.createElement("span");
    type.textContent = event.event_type;
    const summary = document.createElement("strong");
    summary.textContent = event.summary;
    const time = document.createElement("time");
    time.textContent = new Date(event.created_at).toLocaleTimeString();
    item.append(type, summary, time);
    eventList.append(item);
  }
}

function handlePresentationEvents(events) {
  for (const event of events) {
    if (event.event_type === "presentation_command" && event.payload?.voice === "orchestrator") {
      enqueueOrchestratorSpeech(event.event_id);
    }
  }
}

function enqueueOrchestratorSpeech(eventId) {
  if (spokenEventIds.has(eventId) || speechInFlight.has(eventId) || queuedSpeechEventIds.has(eventId)) {
    return;
  }

  queuedSpeechEventIds.add(eventId);
  speechQueue.push(eventId);
  drainSpeechQueue().catch(() => setCameraStatus("Voice playback failed"));
}

async function drainSpeechQueue() {
  if (speechQueueActive) {
    return;
  }

  speechQueueActive = true;
  try {
    while (speechQueue.length > 0) {
      const eventId = speechQueue.shift();
      queuedSpeechEventIds.delete(eventId);
      await playOrchestratorSpeech(eventId);
    }
  } finally {
    speechQueueActive = false;
  }
}

async function playOrchestratorSpeech(eventId) {
  if (spokenEventIds.has(eventId) || speechInFlight.has(eventId)) {
    return;
  }

  speechInFlight.add(eventId);
  let audioUrl = null;
  let audio = null;
  try {
    const response = await fetch(`/api/voice/orchestrator/${eventId}`);
    if (!response.ok) {
      throw new Error("Voice request failed.");
    }
    audioUrl = URL.createObjectURL(await response.blob());
    audio = new Audio(audioUrl);
    currentSpeechAudio = audio;
    await playAudioToEnd(audio);
    spokenEventIds.add(eventId);
  } finally {
    if (currentSpeechAudio === audio) {
      currentSpeechAudio = null;
    }
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    speechInFlight.delete(eventId);
  }
}

function playAudioToEnd(audio) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("error", handleError);
    };
    const handleEnded = () => {
      cleanup();
      resolve();
    };
    const handleError = () => {
      cleanup();
      reject(new Error("Voice playback failed."));
    };

    audio.addEventListener("ended", handleEnded, { once: true });
    audio.addEventListener("error", handleError, { once: true });
    audio.play().catch((error) => {
      cleanup();
      reject(error);
    });
  });
}

function renderSnapshot(snapshot) {
  const { state } = snapshot;
  text("[data-hand-id]", state.hand_id);
  text("[data-street]", state.street);
  text("[data-pot]", String(state.pot));
  text("[data-waiting]", state.waiting_for?.type ?? "none");
  text("[data-board-source]", state.board_source);
  text("[data-engine-status]", state.automation_status);
  text("[data-active-seat]", `S${state.active_player_seat}`);
  text("[data-waiting-reason]", state.waiting_for?.reason ?? "none");
  text("[data-to-call]", String(state.current_bet_to_call));
  text("[data-min-raise]", String(state.min_raise_to));

  renderSeats(state);
  renderCards(document.querySelector("[data-board-row]"), state.board, "No board cards recognised");
  renderRecognition(state.board_recognition);
  renderPrivateStates(snapshot.private_states);
  renderClients(snapshot.client_statuses);
  renderActions(state.legal_actions);
  renderUncertainties(state.uncertainties);
  renderEvents(snapshot.events);
  if (hasRenderedInitialSnapshot || shouldPlayInitialSnapshotSpeech) {
    handlePresentationEvents(snapshot.events);
  } else {
    for (const event of snapshot.events) {
      if (event.event_type === "presentation_command" && event.payload?.voice === "orchestrator") {
        spokenEventIds.add(event.event_id);
      }
    }
  }
  hasRenderedInitialSnapshot = true;
  shouldPlayInitialSnapshotSpeech = false;
  updateBoardFrameSubmission(state.waiting_for?.type === "public_board_cards");

  if (apiStatus) {
    apiStatus.textContent = `${state.hand_id} ${state.automation_status}`;
    apiStatus.classList.add("is-ready");
  }
}

function connectEvents() {
  const protocol = globalThis.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${globalThis.location.host}/ws/events`);

  socket.addEventListener("open", () => {
    if (apiStatus) {
      apiStatus.textContent = "WebSocket live";
      apiStatus.classList.add("is-ready");
    }
  });

  socket.addEventListener("message", (message) => {
    const snapshot = JSON.parse(message.data);
    if (snapshot.type === "snapshot") {
      renderSnapshot(snapshot);
    }
  });

  socket.addEventListener("close", () => {
    if (apiStatus) {
      apiStatus.textContent = "WebSocket reconnecting";
      apiStatus.classList.remove("is-ready");
    }
    globalThis.setTimeout(connectEvents, 1500);
  });
}

async function submitGameControl(action) {
  if (action === "start") {
    shouldPlayInitialSnapshotSpeech = true;
  }
  const response = await fetch(`/api/game/${action}`, { method: "POST" });

  if (!response.ok) {
    throw new Error("Game control failed.");
  }

  const payload = await response.json();
  handlePresentationEvents(payload.events);
  if (apiStatus) {
    apiStatus.textContent = payload.reason;
  }
}

async function loadCameraDevices() {
  if (!cameraDevice || !navigator.mediaDevices?.enumerateDevices) {
    setCameraStatus("Browser camera API unavailable");
    return;
  }

  const selectedDevice = cameraDevice.value;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === "videoinput");
  cameraDevice.replaceChildren();
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "Default camera";
  cameraDevice.append(defaultOption);
  for (const [index, device] of cameras.entries()) {
    const option = document.createElement("option");
    option.value = device.deviceId;
    option.textContent = device.label || `Camera ${index + 1}`;
    cameraDevice.append(option);
  }
  cameraDevice.value = selectedDevice;
  if (cameras.length === 0) {
    setCameraStatus("No browser cameras found");
  } else {
    setCameraStatus(cameraStream ? "Camera ready" : "Camera idle");
  }
}

async function ensureCameraStream() {
  if (!cameraPreview || !navigator.mediaDevices?.getUserMedia) {
    setCameraStatus("Browser camera API unavailable");
    return;
  }

  if (cameraStream) {
    return;
  }

  const deviceId = cameraDevice?.value;
  const constraints = {
    video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "environment" },
    audio: false,
  };
  cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
  cameraPreview.srcObject = cameraStream;
  cameraPreview.hidden = false;
  setCameraStatus("Camera ready");
}

async function chooseCameraDevice() {
  stopCameraStream();
  await ensureCameraStream();
  await loadCameraDevices();
}

function stopCameraStream() {
  if (!cameraStream) {
    return;
  }
  for (const track of cameraStream.getTracks()) {
    track.stop();
  }
  cameraStream = null;
}

function updateBoardFrameSubmission(enabled) {
  shouldSubmitBoardFrames = enabled;
  if (enabled) {
    startFrameLoop();
  } else {
    stopFrameLoop();
  }
}

function startFrameLoop() {
  if (frameTimer) {
    return;
  }
  setCameraStatus("Submitting board frames");
  frameTimer = globalThis.setInterval(() => {
    submitBoardFrame().catch(() => setCameraStatus("Frame submission failed"));
  }, FRAME_INTERVAL_MS);
  submitBoardFrame().catch(() => setCameraStatus("Frame submission failed"));
}

function stopFrameLoop() {
  if (!frameTimer) {
    return;
  }
  globalThis.clearInterval(frameTimer);
  frameTimer = null;
  if (cameraStream) {
    setCameraStatus("Camera ready");
  }
}

async function submitBoardFrame() {
  if (!shouldSubmitBoardFrames || frameInFlight || !cameraPreview) {
    return;
  }

  await ensureCameraStream();
  if (cameraPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }

  frameInFlight = true;
  try {
    const dataUri = captureVideoFrame(cameraPreview);
    const response = await fetch("/api/vision/public-board/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: cameraDevice?.selectedOptions?.[0]?.textContent || "dashboard_browser_camera",
        data_uri: dataUri,
      }),
    });
    const payload = await response.json();
    setCameraStatus(payload.reason ?? (response.ok ? "Frame processed" : "Frame rejected"));
  } finally {
    frameInFlight = false;
  }
}

function captureVideoFrame(video) {
  const scale = FRAME_WIDTH / Math.max(video.videoWidth, 1);
  const canvas = document.createElement("canvas");
  canvas.width = FRAME_WIDTH;
  canvas.height = Math.round(video.videoHeight * scale);
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL(FRAME_MIME_TYPE);
}

function setCameraStatus(value) {
  if (cameraStatus) {
    cameraStatus.textContent = value;
  }
}

for (const control of gameControls) {
  control.addEventListener("click", () => {
    control.disabled = true;
    submitGameControl(control.dataset.gameControl)
      .catch(() => {
        if (apiStatus) {
          apiStatus.textContent = "Control failed";
        }
      })
      .finally(() => {
        control.disabled = false;
      });
  });
}

if (cameraForm) {
  cameraForm.addEventListener("submit", (event) => {
    event.preventDefault();
    chooseCameraDevice().catch(() => setCameraStatus("Camera start failed"));
  });
}

await loadCameraDevices().catch(() => setCameraStatus("Camera permission needed"));
connectEvents();
