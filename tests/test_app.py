from fastapi.testclient import TestClient

from pokerbot_3000.app.server import create_app


def test_index_renders_starter_page():
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "Pokerbot 3000" in response.text
    assert "/static/styles.css" in response.text
    assert "/static/app.js" in response.text


def test_missing_static_file_returns_not_found():
    client = TestClient(create_app())

    response = client.get("/static/missing.css")

    assert response.status_code == 404
