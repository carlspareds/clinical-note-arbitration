"""Token pricing and cost calculation for supported LLM models."""

from typing import Dict, Tuple

# Pricing in USD per 1,000,000 tokens: (input_price_per_1m, output_price_per_1m)
MODEL_PRICING_PER_1M: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-opus-20240229": (15.00, 75.00),
    # Google Gemini
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-pro-latest": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-latest": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    # OpenRouter / DigitalOcean Gradient / Open Source
    "llama3.3-70b-instruct": (0.59, 0.79),
    "llama3.2": (0.00, 0.00),
    "llama3.2:latest": (0.00, 0.00),
    "alibaba-qwen3-32b": (0.30, 0.60),
    "openai-gpt-oss-120b": (0.80, 1.60),
    "deepseek-r1-distill-llama-70b": (0.70, 0.90),
    "deepseek-v4.1-flash": (0.14, 0.28),
    "glm-5.3-flash": (0.10, 0.20),
    "qwen3.8-flash": (0.10, 0.20),
    # Mock / Local
    "mock": (0.00, 0.00),
    "mock-judge": (0.00, 0.00),
    "mock-generator": (0.00, 0.00),
}


def calculate_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate inference cost in USD for given token counts."""
    # Normalize model name
    normalized = model_name.lower().strip()
    if "/" in normalized:
        # e.g. openai/gpt-4o -> gpt-4o
        normalized = normalized.split("/")[-1]

    # Find closest match or default fallback
    pricing = None
    if normalized in MODEL_PRICING_PER_1M:
        pricing = MODEL_PRICING_PER_1M[normalized]
    else:
        for known_model, p in MODEL_PRICING_PER_1M.items():
            if known_model in normalized or normalized in known_model:
                pricing = p
                break

    if pricing is None:
        # Conservative default: $1.00 input / $3.00 output per 1M
        pricing = (1.00, 3.00)

    input_cost = (prompt_tokens / 1_000_000.0) * pricing[0]
    output_cost = (completion_tokens / 1_000_000.0) * pricing[1]
    return round(input_cost + output_cost, 6)
