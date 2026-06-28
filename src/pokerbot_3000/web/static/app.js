const apiStatus = document.querySelector("[data-api-status]");
const eventList = document.querySelector("[data-event-list]");
const gameControls = document.querySelectorAll("[data-game-control]");

async function refreshApiStatus() {
  if (!apiStatus) {
    return;
  }

  try {
    const [stateResponse, eventsResponse] = await Promise.all([fetch("/api/state"), fetch("/api/events?limit=5")]);

    if (!stateResponse.ok || !eventsResponse.ok) {
      throw new Error("Orchestrator API returned an error.");
    }

    const state = await stateResponse.json();
    const events = await eventsResponse.json();
    apiStatus.textContent = `${state.hand_id} ${state.automation_status}`;
    apiStatus.classList.add("is-ready");

    if (eventList) {
      eventList.dataset.loadedEvents = String(events.length);
    }
  } catch {
    apiStatus.textContent = "API unavailable";
  }
}

async function submitGameControl(action) {
  const response = await fetch(`/api/game/${action}`, { method: "POST" });

  if (!response.ok) {
    throw new Error("Game control failed.");
  }

  globalThis.location.reload();
}

for (const control of gameControls) {
  control.addEventListener("click", () => {
    control.disabled = true;
    submitGameControl(control.dataset.gameControl).catch(() => {
      control.disabled = false;
      if (apiStatus) {
        apiStatus.textContent = "Control failed";
      }
    });
  });
}

await refreshApiStatus();
