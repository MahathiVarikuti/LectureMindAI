import pytest

from tests.conftest import make_pdf


@pytest.mark.slow
def test_real_minilm_pdf_query_smoke(client, session_a, monkeypatch):
    monkeypatch.undo()
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        return "The answer is in the lecture [1]."

    monkeypatch.setattr("app.services.llm_service.chat", fake_chat)
    response = client.post(
        "/upload",
        data={"session_id": session_a},
        files={"file": ("tiny.pdf", make_pdf("Gradient descent follows the negative gradient."), "application/pdf")},
    )
    assert response.status_code == 200
    answer = client.post(
        "/query",
        json={"session_id": session_a, "query": "What does gradient descent follow?"},
    )
    assert answer.status_code == 200
    assert calls
