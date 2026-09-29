"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from arbiter.api import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "supported_providers" in data


def test_api_dashboard():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Clinical Note Arbitration" in response.text


def test_api_samples():
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) > 0
    assert "transcript" in samples[0]


def test_api_arbitrate(sample_transcript: str):
    payload = {
        "transcript": sample_transcript,
        "note_a": "SUBJECTIVE: Patient with Lisinopril cough and Penicillin allergy.\nPLAN: Stop Lisinopril.",
        "note_b": "SUBJECTIVE: Patient with cough.\nPLAN: Give Amoxicillin.",
        "judge_model": "mock-judge",
    }
    response = client.post("/api/arbitrate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "final_verdict" in data
    assert "eval_ab" in data
    assert "eval_ba" in data


def test_api_score(sample_transcript: str):
    payload = {
        "transcript": sample_transcript,
        "note": "SUBJECTIVE: Cough on Lisinopril.\nPLAN: Switch to Losartan.",
        "judge_model": "mock-judge",
    }
    response = client.post("/api/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "overall_score" in data
    assert "criterion_scores" in data


def test_api_generate(sample_transcript: str):
    payload = {
        "transcript": sample_transcript,
        "model_a": "mock-generator",
        "model_b": "mock-generator",
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "note_a" in data
    assert "note_b" in data
