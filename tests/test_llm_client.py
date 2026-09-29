"""Unit tests for Unified LLM client, providers, and cost calculations."""

from arbiter.llm.client import get_llm_client
from arbiter.llm.pricing import calculate_cost_usd
from arbiter.llm.providers import MockClinicalLLMProvider, extract_json_from_text


def test_calculate_cost_usd():
    # Test OpenAI GPT-4o pricing ($2.50 input, $10.00 output per 1M)
    cost = calculate_cost_usd("gpt-4o", prompt_tokens=10_000, completion_tokens=1_000)
    assert cost > 0.0
    assert abs(cost - (0.025 + 0.010)) < 1e-4

    # Test local / mock model (free)
    mock_cost = calculate_cost_usd("mock", 5000, 500)
    assert mock_cost == 0.0

    ollama_cost = calculate_cost_usd("llama3.2", 5000, 500)
    assert ollama_cost == 0.0


def test_extract_json_from_text():
    # Direct JSON
    assert extract_json_from_text('{"score": 95}') == {"score": 95}

    # Markdown fenced code block
    fenced = 'Here is the result:\n```json\n{\n  "status": "ok",\n  "count": 3\n}\n```\nDone.'
    assert extract_json_from_text(fenced) == {"status": "ok", "count": 3}

    # Unfenced text with embedded object
    embedded = 'Analysis complete: {"winner": "Option 1"} thanks.'
    assert extract_json_from_text(embedded) == {"winner": "Option 1"}

    # Invalid JSON
    assert extract_json_from_text("No json here") is None


def test_mock_clinical_llm_provider():
    provider = MockClinicalLLMProvider()
    resp = provider.generate("Evaluate the following clinical note:", json_mode=True)
    assert resp.parsed_json is not None
    assert "overall_score" in resp.parsed_json
    assert resp.provider == "mock"


def test_get_llm_client_mock():
    client = get_llm_client("mock-judge")
    assert client.model_identifier == "mock-judge"
    resp = client.generate("Option 1 vs Option 2 comparison")
    assert resp.parsed_json is not None
