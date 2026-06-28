from fastapi.testclient import TestClient

from pokerbot_3000.app.server import create_app


def test_index_renders_starter_page():
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "Pokerbot 3000" in response.text
    assert "Table State" in response.text
    assert "/static/styles.css" in response.text
    assert "/static/app.js" in response.text


def test_missing_static_file_returns_not_found():
    client = TestClient(create_app())

    response = client.get("/static/missing.css")

    assert response.status_code == 404


def test_api_state_returns_public_game_snapshot():
    client = TestClient(create_app())

    response = client.get("/api/state")

    assert response.status_code == 200
    payload = response.json()
    assert payload["hand_id"] == "hand_001"
    assert payload["automation_status"] == "stopped"
    assert payload["waiting_for"] is None
    assert payload["players"]["1"]["name"] == "Che"
    assert "reachy" not in payload


def test_api_events_returns_initial_event():
    client = TestClient(create_app())

    response = client.get("/api/events")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["event_type"] == "system"


def test_api_start_and_stop_game():
    client = TestClient(create_app())

    start_response = client.post("/api/game/start")

    assert start_response.status_code == 200
    start_payload = start_response.json()
    assert start_payload["accepted"] is True
    assert start_payload["state"]["automation_status"] == "waiting_for_external_input"
    assert start_payload["state"]["waiting_for"]["type"] == "human_action"

    stop_response = client.post("/api/game/stop")

    assert stop_response.status_code == 200
    stop_payload = stop_response.json()
    assert stop_payload["accepted"] is True
    assert stop_payload["state"]["automation_status"] == "stopped"
    assert stop_payload["state"]["waiting_for"] is None


def test_api_human_action_advances_until_eliza_input_needed():
    client = TestClient(create_app())
    client.post("/api/game/start")

    response = client.post(
        "/api/inputs/human-action",
        json={
            "source": "voice",
            "action": {"type": "bet", "amount": 100, "unit": "chips"},
            "raw_transcript": "bet one hundred",
            "confidence": 0.95,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["state"]["pot"] == 100
    assert payload["state"]["waiting_for"]["type"] == "private_cards"
    assert payload["state"]["waiting_for"]["agent_id"] == "eliza"


def test_api_thin_client_private_cards_trigger_internal_agent_turn():
    client = TestClient(create_app())
    client.post("/api/game/start")
    client.post(
        "/api/inputs/human-action",
        json={"source": "voice", "action": {"type": "bet", "amount": 100}},
    )

    response = client.post(
        "/api/clients/eliza/private-cards",
        json={
            "agent_id": "eliza",
            "seat": 3,
            "hole_cards": [
                {"rank": "9", "suit": "clubs"},
                {"rank": "9", "suit": "diamonds"},
            ],
            "source": "eliza_browser_webcam",
            "confidence": 0.89,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["state"]["pot"] == 200
    assert payload["state"]["waiting_for"]["agent_id"] == "reachy"
    assert "agent_decision" in {event["event_type"] for event in payload["events"]}
