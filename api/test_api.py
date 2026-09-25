from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():

    response = client.get(
        "/api/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "supportiq-api"
    assert "ready" in data


def test_chat_empty_query():

    response = client.post(
        "/api/chat",
        json={
            "query": ""
        },
    )

    assert response.status_code == 422


def test_chat_top_k_too_large():

    response = client.post(
        "/api/chat",
        json={
            "query": "Where is my order?",
            "top_k": 20,
        },
    )

    assert response.status_code == 422