const clientId = document.body.dataset.clientId || "eliza";
const face = document.querySelector("[data-eliza-face]");
const speech = document.querySelector("[data-eliza-speech]");
const connection = document.querySelector("[data-client-connection]");
const waiting = document.querySelector("[data-client-waiting]");
const cameraDevice = document.querySelector("[data-private-camera-device]");
const cameraPreview = document.querySelector("[data-private-camera-preview]");
const cardStatus = document.querySelector("[data-card-status]");
const eventList = document.querySelector("[data-client-events]");

const DEFAULT_FACE = "🙂";
const FRAME_WIDTH = 1600;
const FRAME_MIME_TYPE = "image/png";
const ANTI_FLICKER_FRAME_RATE = 25;
const expressedEventIds = new Set();
let cameraStream = null;
let latestSnapshot = null;
let faceResetTimer = null;
let captureInFlight = false;
let submittedCaptureKey = null;
let captureRetryTimer = null;
let observedHandId = null;

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
  const handId = snapshot.state?.hand_id ?? null;
  if (handId !== observedHandId || snapshot.state?.automation_status === "stopped") {
    observedHandId = handId;
    submittedCaptureKey = null;
  }
  const waitingFor = snapshot.state?.waiting_for;
  const privateState = snapshot.private_states?.[0];
  const shouldCapture =
    waitingFor?.type === "private_cards" && waitingFor.agent_id === clientId && !hasCompletePrivateCards(privateState);
  if (!shouldCapture && captureRetryTimer) {
    globalThis.clearTimeout(captureRetryTimer);
    captureRetryTimer = null;
  }
  setText(waiting, shouldCapture ? "show cards" : (waitingFor?.type ?? "idle"));
  setText(cardStatus, privateCardStatus(privateState, shouldCapture));
  renderEvents(snapshot.events ?? []);
  handlePresentationEvents(snapshot.events ?? []);
  maybeAutoCapturePrivateCards(shouldCapture);
}

function hasCompletePrivateCards(privateState) {
  return (privateState?.hole_cards?.length ?? 0) >= 2;
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
  if (!cameraDevice) {
    return;
  }
  if (!navigator.mediaDevices?.enumerateDevices) {
    setText(cardStatus, cameraUnavailableMessage());
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
    setText(cardStatus, "no cameras found");
  } else {
    setText(cardStatus, cameraStream ? "camera ready" : "camera idle");
  }
}

async function ensureCameraStream() {
  if (!cameraPreview || !navigator.mediaDevices?.getUserMedia) {
    setText(cardStatus, cameraUnavailableMessage());
    return;
  }
  if (cameraStream) {
    return;
  }
  const deviceId = cameraDevice?.value;
  cameraStream = await getCameraStream(deviceId);
  cameraPreview.srcObject = cameraStream;
  await loadCameraDevices();
}

async function getCameraStream(deviceId) {
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: videoConstraints(deviceId, { frameRate: { ideal: ANTI_FLICKER_FRAME_RATE, max: ANTI_FLICKER_FRAME_RATE } }),
      audio: false,
    });
  } catch (error) {
    if (error?.name !== "OverconstrainedError") {
      throw error;
    }
    return navigator.mediaDevices.getUserMedia({
      video: videoConstraints(deviceId),
      audio: false,
    });
  }
}

function videoConstraints(deviceId, extras = {}) {
  return {
    ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "environment" }),
    ...extras,
  };
}

function cameraUnavailableMessage() {
  return globalThis.isSecureContext ? "camera unavailable" : "HTTPS or localhost required";
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
  submittedCaptureKey = null;
  await ensureCameraStream();
}

function privateCardCaptureKey() {
  const state = latestSnapshot?.state;
  const waitingFor = state?.waiting_for;
  if (hasCompletePrivateCards(latestSnapshot?.private_states?.[0])) {
    return null;
  }
  if (waitingFor?.type !== "private_cards" || waitingFor.agent_id !== clientId) {
    return null;
  }
  return `${state.hand_id}:${waitingFor.agent_id}:${waitingFor.type}`;
}

function maybeAutoCapturePrivateCards(shouldCapture) {
  if (!shouldCapture || captureInFlight) {
    return;
  }
  const captureKey = privateCardCaptureKey();
  if (!captureKey || submittedCaptureKey === captureKey) {
    return;
  }
  capturePrivateCards().catch(() => {
    submittedCaptureKey = null;
    setText(cardStatus, "capture failed");
  });
}

function retryAutoCapture() {
  if (captureRetryTimer) {
    return;
  }
  captureRetryTimer = globalThis.setTimeout(() => {
    captureRetryTimer = null;
    const waitingFor = latestSnapshot?.state?.waiting_for;
    maybeAutoCapturePrivateCards(waitingFor?.type === "private_cards" && waitingFor.agent_id === clientId);
  }, 700);
}

async function capturePrivateCards() {
  const captureKey = privateCardCaptureKey();
  if (!captureKey || submittedCaptureKey === captureKey) {
    return;
  }
  captureInFlight = true;
  submittedCaptureKey = captureKey;
  setText(cardStatus, "reading");
  try {
    await ensureCameraStream();
    if (cameraPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      setText(cardStatus, "camera warming");
      submittedCaptureKey = null;
      retryAutoCapture();
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
    if (!response.ok || !payload.accepted) {
      submittedCaptureKey = null;
      retryAutoCapture();
    }
  } finally {
    captureInFlight = false;
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

setText(face, DEFAULT_FACE);
await loadCameraDevices().catch(() => setText(cardStatus, "camera unavailable"));
await ensureCameraStream().catch(() => setText(cardStatus, "camera permission needed"));
connectEvents();
