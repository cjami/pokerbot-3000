const clientId = document.body.dataset.clientId || "eliza";
const face = document.querySelector("[data-eliza-face]");
const speech = document.querySelector("[data-eliza-speech]");
const connection = document.querySelector("[data-client-connection]");
const waiting = document.querySelector("[data-client-waiting]");
const cameraDevice = document.querySelector("[data-private-camera-device]");
const cameraPreview = document.querySelector("[data-private-camera-preview]");
const captureButton = document.querySelector("[data-capture-private-cards]");
const cardStatus = document.querySelector("[data-card-status]");
const eventList = document.querySelector("[data-client-events]");

const DEFAULT_FACE = "🙂";
const FRAME_WIDTH = 1600;
const FRAME_MIME_TYPE = "image/png";
const expressedEventIds = new Set();
let cameraStream = null;
let latestSnapshot = null;
let faceResetTimer = null;

const emojiByEmotion = {
  calm: "🙂",
  confident: "😎",
  celebrate: "🥳",
  confused: "🤔",
  sad: "🙁",
};

function setText(element, value) {
  if (element) {
    element.textContent = value;
  }
}

async function postStatus(connectionState, status, detail = null) {
  await fetch(`/api/clients/${clientId}/status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ connection: connectionState, status, detail }),
  }).catch(() => {});
}

function connectEvents() {
  const protocol = globalThis.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${globalThis.location.host}/ws/clients/${clientId}`);

  socket.addEventListener("open", () => {
    setText(connection, "connected");
    postStatus("connected", "Eliza browser connected").catch(() => {});
  });

  socket.addEventListener("message", (message) => {
    const snapshot = JSON.parse(message.data);
    if (snapshot.type === "snapshot") {
      renderSnapshot(snapshot);
    }
  });

  socket.addEventListener("close", () => {
    setText(connection, "reconnecting");
    postStatus("disconnected", "Eliza browser reconnecting").catch(() => {});
    globalThis.setTimeout(connectEvents, 1500);
  });
}

function renderSnapshot(snapshot) {
  latestSnapshot = snapshot;
  const waitingFor = snapshot.state?.waiting_for;
  const shouldCapture = waitingFor?.type === "private_cards" && waitingFor.agent_id === clientId;
  captureButton.disabled = !shouldCapture;
  setText(waiting, shouldCapture ? "show cards" : (waitingFor?.type ?? "idle"));
  setText(cardStatus, privateCardStatus(snapshot.private_states?.[0], shouldCapture));
  renderEvents(snapshot.events ?? []);
  handlePresentationEvents(snapshot.events ?? []);
}

function privateCardStatus(privateState, shouldCapture) {
  if (shouldCapture) {
    return "requested";
  }
  if (!privateState) {
    return "unknown";
  }
  return `${privateState.hole_cards.length}/2 cards`;
}

function renderEvents(events) {
  if (!eventList) {
    return;
  }
  eventList.replaceChildren();
  const relevant = events
    .filter((event) => event.source === `agent:${clientId}` || event.payload?.target_client === clientId)
    .slice(-8)
    .reverse();
  for (const event of relevant) {
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
    if (event.event_type !== "presentation_command" || event.payload?.target_client !== clientId) {
      continue;
    }
    if (expressedEventIds.has(event.event_id)) {
      continue;
    }
    expressedEventIds.add(event.event_id);
    express(event).catch(() => {
      setText(speech, event.payload?.speech ?? "...");
      showEmotion(event.payload?.emotion);
    });
  }
}

async function express(event) {
  const line = event.payload?.speech ?? "";
  setText(speech, line || "...");
  showEmotion(event.payload?.emotion);
  if (!line) {
    return;
  }
  const response = await fetch(`/api/voice/eliza/${event.event_id}`);
  if (!response.ok) {
    return;
  }
  const audioUrl = URL.createObjectURL(await response.blob());
  try {
    await playAudio(audioUrl);
  } finally {
    URL.revokeObjectURL(audioUrl);
  }
}

function showEmotion(emotion) {
  setText(face, emojiByEmotion[emotion] ?? DEFAULT_FACE);
  if (faceResetTimer) {
    globalThis.clearTimeout(faceResetTimer);
  }
  faceResetTimer = globalThis.setTimeout(() => setText(face, DEFAULT_FACE), 2200);
}

function playAudio(audioUrl) {
  return new Promise((resolve, reject) => {
    const audio = new Audio(audioUrl);
    audio.addEventListener("ended", resolve, { once: true });
    audio.addEventListener("error", () => reject(new Error("Audio playback failed.")), { once: true });
    audio.play().catch(reject);
  });
}

async function loadCameraDevices() {
  if (!cameraDevice || !navigator.mediaDevices?.enumerateDevices) {
    setText(cardStatus, "camera unavailable");
    return;
  }
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
}

async function ensureCameraStream() {
  if (!cameraPreview || !navigator.mediaDevices?.getUserMedia) {
    setText(cardStatus, "camera unavailable");
    return;
  }
  if (cameraStream) {
    return;
  }
  const deviceId = cameraDevice?.value;
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "environment" },
    audio: false,
  });
  cameraPreview.srcObject = cameraStream;
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

async function chooseCameraDevice() {
  stopCameraStream();
  await ensureCameraStream();
}

async function capturePrivateCards() {
  if (!latestSnapshot?.state?.waiting_for || captureButton.disabled) {
    return;
  }
  captureButton.disabled = true;
  setText(cardStatus, "reading");
  try {
    await ensureCameraStream();
    if (cameraPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      setText(cardStatus, "camera warming");
      return;
    }
    const response = await fetch(`/api/clients/${clientId}/private-cards/frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: "eliza_browser_webcam",
        data_uri: captureVideoFrame(cameraPreview),
      }),
    });
    const payload = await response.json();
    setText(cardStatus, payload.reason ?? (response.ok ? "submitted" : "rejected"));
  } finally {
    captureButton.disabled = false;
  }
}

function captureVideoFrame(video) {
  const canvas = document.createElement("canvas");
  const scale = FRAME_WIDTH / Math.max(video.videoWidth, 1);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL(FRAME_MIME_TYPE);
}

cameraDevice?.addEventListener("change", () => {
  chooseCameraDevice().catch(() => setText(cardStatus, "camera failed"));
});
captureButton?.addEventListener("click", () => {
  capturePrivateCards().catch(() => setText(cardStatus, "capture failed"));
});

setText(face, DEFAULT_FACE);
await ensureCameraStream().catch(() => setText(cardStatus, "camera permission needed"));
connectEvents();
