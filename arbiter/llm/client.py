"""Unified LLM client interface supporting OpenAI, Anthropic, Gemini, OpenRouter, Gradient, Ollama, and Mock."""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

from arbiter.llm.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    GeminiProvider,
    LLMResponse,
    MockClinicalLLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
)

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger("arbiter.llm")


class UnifiedLLMClient:
    """Provider-agnostic LLM interface for clinical note generation, scoring, and arbitration."""

    def __init__(self, model_identifier: str, provider: BaseLLMProvider):
        self.model_identifier = model_identifier
        self.provider = provider

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Execute prompt against underlying provider."""
        return self.provider.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_mode=json_mode,
        )


def get_llm_client(
    model_identifier: Optional[str] = None,
    timeout: float = 60.0,
) -> UnifiedLLMClient:
    """
    Factory function returning a unified LLM client instance for any supported model or provider.
    Fallback priority:
      1. Explicit provider prefix (e.g. 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet', 'ollama/llama3.2')
      2. Environment API keys (DO_GRADIENT_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY)
      3. Local Ollama instance
      4. MockClinicalLLMProvider (zero-cost offline testing)
    """
    if not model_identifier:
        model_identifier = os.getenv("DEFAULT_JUDGE_MODEL", "mock-judge")

    model_lower = model_identifier.lower().strip()

    # Check for Mock provider
    if model_lower.startswith("mock") or model_lower in ["mock-judge", "mock-generator", "mock"]:
        return UnifiedLLMClient(
            model_identifier, MockClinicalLLMProvider(model_identifier, timeout=timeout)
        )

    # Check for explicit Ollama prefix
    if model_lower.startswith("ollama/") or model_lower in ["llama3.2", "llama3.2:latest"]:
        model_name = model_identifier.replace("ollama/", "")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return UnifiedLLMClient(
            model_identifier,
            OllamaProvider(model_name=model_name, base_url=base_url, timeout=timeout),
        )

    # Check for OpenCode / Custom OpenAI-compatible endpoint
    if model_lower.startswith("opencode/") or model_lower in [
        "deepseek-v4.1-flash",
        "glm-5.3-flash",
        "qwen3.8-flash",
    ]:
        model_name = model_identifier.replace("opencode/", "")
        opencode_key = (
            os.getenv("OPENCODE_API_KEY")
            or os.getenv("OPENCODE_GO_API_KEY")
            or os.getenv("CUSTOM_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DO_GRADIENT_API_KEY", "")
        )
        opencode_base = os.getenv("CUSTOM_OPENAI_BASE_URL", "https://opencode.ai/zen/go/v1")
        return UnifiedLLMClient(
            model_identifier,
            OpenAICompatibleProvider(
                model_name=model_name,
                api_key=opencode_key,
                base_url=opencode_base,
                provider_name="opencode",
                timeout=timeout,
            ),
        )

    # Check for DigitalOcean Gradient / custom OpenAI base
    if model_lower.startswith("gradient/") or "gradient" in model_lower:
        model_name = model_identifier.replace("gradient/", "")
        gradient_key = os.getenv("DO_GRADIENT_API_KEY", "")
        gradient_base = os.getenv("CUSTOM_OPENAI_BASE_URL", "https://inference.do-ai.run/v1")
        return UnifiedLLMClient(
            model_identifier,
            OpenAICompatibleProvider(
                model_name=model_name,
                api_key=gradient_key,
                base_url=gradient_base,
                provider_name="gradient",
                timeout=timeout,
            ),
        )

    # Check for OpenRouter
    if model_lower.startswith("openrouter/"):
        model_name = model_identifier.replace("openrouter/", "")
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        return UnifiedLLMClient(
            model_identifier,
            OpenAICompatibleProvider(
                model_name=model_name,
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                provider_name="openrouter",
                timeout=timeout,
            ),
        )

    # Check for Anthropic
    if model_lower.startswith("anthropic/") or "claude" in model_lower:
        model_name = model_identifier.replace("anthropic/", "")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            logger.warning("ANTHROPIC_API_KEY not found; falling back to Mock provider.")
            return UnifiedLLMClient(
                model_identifier, MockClinicalLLMProvider(model_identifier, timeout=timeout)
            )
        return UnifiedLLMClient(
            model_identifier,
            AnthropicProvider(model_name=model_name, api_key=anthropic_key, timeout=timeout),
        )

    # Check for Gemini
    if model_lower.startswith("gemini/") or "gemini" in model_lower:
        model_name = model_identifier.replace("gemini/", "")
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if not gemini_key:
            logger.warning("GEMINI_API_KEY not found; falling back to Mock provider.")
            return UnifiedLLMClient(
                model_identifier, MockClinicalLLMProvider(model_identifier, timeout=timeout)
            )
        return UnifiedLLMClient(
            model_identifier,
            GeminiProvider(model_name=model_name, api_key=gemini_key, timeout=timeout),
        )

    # Check for OpenAI
    if model_lower.startswith("openai/") or "gpt-" in model_lower:
        model_name = model_identifier.replace("openai/", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            # Check if gradient key is available as alternate OpenAI endpoint
            gradient_key = os.getenv("DO_GRADIENT_API_KEY", "")
            if gradient_key:
                logger.info("Using DO_GRADIENT_API_KEY for OpenAI-compatible endpoint.")
                return UnifiedLLMClient(
                    model_identifier,
                    OpenAICompatibleProvider(
                        model_name=model_name,
                        api_key=gradient_key,
                        base_url=os.getenv(
                            "CUSTOM_OPENAI_BASE_URL", "https://inference.do-ai.run/v1"
                        ),
                        provider_name="gradient",
                        timeout=timeout,
                    ),
                )
            logger.warning("OPENAI_API_KEY not found; falling back to Mock provider.")
            return UnifiedLLMClient(
                model_identifier, MockClinicalLLMProvider(model_identifier, timeout=timeout)
            )

        return UnifiedLLMClient(
            model_identifier,
            OpenAICompatibleProvider(
                model_name=model_name,
                api_key=openai_key,
                base_url="https://api.openai.com/v1",
                provider_name="openai",
                timeout=timeout,
            ),
        )

    # Check if DigitalOcean Gradient key exists in env
    gradient_key = os.getenv("DO_GRADIENT_API_KEY", "")
    if gradient_key:
        return UnifiedLLMClient(
            model_identifier,
            OpenAICompatibleProvider(
                model_name=model_identifier,
                api_key=gradient_key,
                base_url=os.getenv("CUSTOM_OPENAI_BASE_URL", "https://inference.do-ai.run/v1"),
                provider_name="gradient",
                timeout=timeout,
            ),
        )

    # Fallback to Mock provider
    logger.info(f"No specific provider matched for {model_identifier}; using Mock provider.")
    return UnifiedLLMClient(
        model_identifier, MockClinicalLLMProvider(model_identifier, timeout=timeout)
    )
