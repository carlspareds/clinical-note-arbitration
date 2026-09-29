"""Provider implementations for OpenAI, Anthropic, Gemini, OpenRouter, Gradient, Ollama, and Mock."""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel

from arbiter.llm.pricing import calculate_cost_usd


class LLMResponse(BaseModel):
    content: str
    parsed_json: Optional[Dict[str, Any]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    model: str
    provider: str


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract and parse JSON from LLM text responses."""
    text = text.strip()
    # 1. Try direct json parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. Extract ```json ... ``` code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except Exception:
            pass

    # 3. Extract outermost { ... }
    bracket_match = re.search(r"(\{[\s\S]*\})", text)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(1).strip())
        except Exception:
            pass

    return None


class BaseLLMProvider(ABC):
    def __init__(self, model_name: str, timeout: float = 60.0):
        self.model_name = model_name
        self.timeout = timeout

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        pass


class OpenAICompatibleProvider(BaseLLMProvider):
    """Handles OpenAI, OpenRouter, DigitalOcean Gradient, and Ollama v1 endpoints."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        provider_name: str = "openai",
        timeout: float = 60.0,
    ):
        super().__init__(model_name, timeout)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.provider_name = provider_name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        start_time = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }
        if self.provider_name == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/carlspareds/clinical-note-arbitration"
            headers["X-Title"] = "Clinical Note Arbitration"
        if "opencode.ai" in self.base_url or os.getenv("OPENCODE_SESSION_ID"):
            headers["x-opencode-session"] = os.getenv(
                "OPENCODE_SESSION_ID", "arbiter-clinical-eval"
            )

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode and self.provider_name in [
            "openai",
            "openrouter",
            "opencode",
            "gradient",
            "custom",
        ]:
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        latency = time.time() - start_time
        choice = data["choices"][0]
        content = choice["message"]["content"] or ""

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(prompt) // 4)
        completion_tokens = usage.get("completion_tokens", len(content) // 4)
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = calculate_cost_usd(self.model_name, prompt_tokens, completion_tokens)

        parsed_json = extract_json_from_text(content) if json_mode else None

        return LLMResponse(
            content=content,
            parsed_json=parsed_json,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_seconds=round(latency, 3),
            model=self.model_name,
            provider=self.provider_name,
        )


class AnthropicProvider(BaseLLMProvider):
    """Handles Anthropic Claude models via Messages API."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com/v1",
        timeout: float = 60.0,
    ):
        super().__init__(model_name, timeout)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        start_time = time.time()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/messages",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

        latency = time.time() - start_time
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", len(prompt) // 4)
        completion_tokens = usage.get("output_tokens", len(content) // 4)
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = calculate_cost_usd(self.model_name, prompt_tokens, completion_tokens)

        parsed_json = extract_json_from_text(content) if json_mode else None

        return LLMResponse(
            content=content,
            parsed_json=parsed_json,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_seconds=round(latency, 3),
            model=self.model_name,
            provider="anthropic",
        )


class GeminiProvider(BaseLLMProvider):
    """Handles Google Gemini models via REST API."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        super().__init__(model_name, timeout)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        start_time = time.time()
        # Clean model name e.g. gemini/gemini-1.5-flash -> gemini-1.5-flash
        model_id = self.model_name.split("/")[-1]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={self.api_key}"

        contents: List[Dict[str, Any]] = []
        if system_prompt:
            contents.append(
                {"role": "user", "parts": [{"text": f"SYSTEM INSTRUCTION: {system_prompt}"}]}
            )
            contents.append(
                {
                    "role": "model",
                    "parts": [
                        {"text": "Understood. I will strictly follow clinical instructions."}
                    ],
                }
            )

        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        latency = time.time() - start_time
        candidates = data.get("candidates", [])
        content = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for p in parts:
                content += p.get("text", "")

        usage_meta = data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", len(prompt) // 4)
        completion_tokens = usage_meta.get("candidatesTokenCount", len(content) // 4)
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = calculate_cost_usd(self.model_name, prompt_tokens, completion_tokens)

        parsed_json = extract_json_from_text(content) if json_mode else None

        return LLMResponse(
            content=content,
            parsed_json=parsed_json,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_seconds=round(latency, 3),
            model=self.model_name,
            provider="gemini",
        )


class OllamaProvider(BaseLLMProvider):
    """Handles local Ollama instances via native API or chat endpoints."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: Optional[str] = None,
        timeout: float = 90.0,
    ):
        super().__init__(model_name, timeout)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip(
            "/"
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        start_time = time.time()
        url = f"{self.base_url}/api/chat"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        latency = time.time() - start_time
        content = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", len(prompt) // 4)
        completion_tokens = data.get("eval_count", len(content) // 4)
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = 0.0  # Local compute

        parsed_json = extract_json_from_text(content) if json_mode else None

        return LLMResponse(
            content=content,
            parsed_json=parsed_json,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_seconds=round(latency, 3),
            model=self.model_name,
            provider="ollama",
        )


class MockClinicalLLMProvider(BaseLLMProvider):
    """
    Mock LLM provider generating clinically realistic SOAP notes, rubric scores,
    and pairwise arbitration verdicts for zero-cost testing and deterministic validation.
    """

    def __init__(self, model_name: str = "mock-judge", timeout: float = 5.0):
        super().__init__(model_name, timeout)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        start_time = time.time()
        prompt_lower = prompt.lower()

        # 1. Check if this is a Pairwise Judge request
        if ("option 1" in prompt_lower and "option 2" in prompt_lower) or (
            "pairwise" in prompt_lower
        ):
            response_dict = self._mock_pairwise_judgment(prompt)
            content = json.dumps(response_dict, indent=2)
            parsed_json = response_dict
        # 2. Check if this is a Single Note Scorer request or HealthBench validation
        elif (
            "single-note" in prompt_lower
            or "evaluate the following clinical note" in prompt_lower
            or "evaluate the above clinical note" in prompt_lower
            or "clinical note to evaluate" in prompt_lower
            or "healthbench" in prompt_lower
            or "physician gold criteria" in prompt_lower
        ):
            response_dict = self._mock_single_note_evaluation(prompt)
            content = json.dumps(response_dict, indent=2)
            parsed_json = response_dict
        # 3. Default: Note generator response
        else:
            content = self._mock_generate_soap_note(prompt)
            parsed_json = extract_json_from_text(content) if json_mode else None

        prompt_tokens = len(prompt) // 4
        completion_tokens = len(content) // 4
        total_tokens = prompt_tokens + completion_tokens
        cost_usd = 0.0

        return LLMResponse(
            content=content,
            parsed_json=parsed_json,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_seconds=round(time.time() - start_time, 4),
            model=self.model_name,
            provider="mock",
        )

    def _mock_pairwise_judgment(self, prompt: str) -> Dict[str, Any]:
        """Generate clinical rationale comparing Option 1 and Option 2 based on note contents."""
        prompt_lower = prompt.lower()

        # Split prompt into option chunks if possible
        opt1_chunk = ""
        opt2_chunk = ""
        if "option 1 clinical note:" in prompt_lower:
            parts = prompt.split("OPTION 1 CLINICAL NOTE:")
            if len(parts) > 1:
                subparts = parts[1].split("OPTION 2 CLINICAL NOTE:")
                opt1_chunk = subparts[0].lower()
                if len(subparts) > 1:
                    opt2_chunk = subparts[1].split("PRESENTATION ORDER:")[0].lower()

        # Helper to detect clinical flaws in a note text chunk
        def extract_option_violations(text: str) -> List[Dict[str, Any]]:
            viols = []
            # Allergy / Penicillin flaw
            if "nkda" in text and "amoxicillin" in text:
                viols.append(
                    {
                        "criterion_id": "ALLERGIES_AND_INTOLERANCES",
                        "severity": "CRITICAL",
                        "description": "Documented NKDA despite severe penicillin anaphylaxis and prescribed Amoxicillin.",
                        "excerpt_quote": "Allergies: NKDA ... Plan: Amoxicillin 500mg TID",
                        "clinical_consequence": "Acute fatal risk of beta-lactam anaphylactic shock.",
                    }
                )
            # Beta blocker doubling in orthostasis
            if "increase metoprolol to 100mg" in text or "metoprolol 100mg" in text:
                viols.append(
                    {
                        "criterion_id": "PLAN_SAFETY_AND_CONTRAINDICATIONS",
                        "severity": "CRITICAL",
                        "description": "Doubled Metoprolol dose in patient with symptomatic orthostatic hypotension and omitted diabetes therapy.",
                        "excerpt_quote": "Increase Metoprolol to 100mg daily",
                        "clinical_consequence": "Severe symptomatic hypotension, syncope, and worsening hyperglycemia.",
                    }
                )
            # Renal colic misdiagnosis as lumbar strain
            if "musculoskeletal lumbar strain" in text or (
                "lumbar" in text and "ibuprofen" in text and "flank" in prompt_lower
            ):
                viols.append(
                    {
                        "criterion_id": "RED_FLAGS_AND_ACUTE_RISKS",
                        "severity": "CRITICAL",
                        "description": "Misdiagnosed acute renal colic with gross hematuria as musculoskeletal strain; discharged without CT imaging.",
                        "excerpt_quote": "Musculoskeletal lumbar strain ... Ibuprofen 400mg",
                        "clinical_consequence": "Delayed diagnosis of obstructive nephrolithiasis, renal injury, and uncontrolled pain.",
                    }
                )
            # Febrile neonate watchful waiting
            if (
                "acetaminophen" in text
                and "48 hours" in text
                and ("neonate" in prompt_lower or "3-week-old" in prompt_lower)
            ):
                viols.append(
                    {
                        "criterion_id": "RED_FLAGS_AND_ACUTE_RISKS",
                        "severity": "CRITICAL",
                        "description": "Advised home antipyretics and watchful waiting for febrile 3-week-old neonate instead of urgent ED sepsis workup.",
                        "excerpt_quote": "Advise the mother to give infant acetaminophen ... call back if fever persists 48 hours",
                        "clinical_consequence": "High risk of neonatal bacterial meningitis, septic shock, and infant mortality.",
                    }
                )
            # CKD gout high-dose NSAID
            if ("indomethacin" in text or "naproxen" in text) and "egfr 22" in prompt_lower:
                viols.append(
                    {
                        "criterion_id": "PLAN_SAFETY_AND_CONTRAINDICATIONS",
                        "severity": "CRITICAL",
                        "description": "Prescribed full-dose NSAID in patient with severe chronic kidney disease (eGFR 22).",
                        "excerpt_quote": "High-dose NSAID",
                        "clinical_consequence": "Precipitation of acute renal failure requiring emergent dialysis.",
                    }
                )
            # Fallback minor violation if note is generally weak
            if not viols and ("return in 3 months" in text or "return in 2 months" in text):
                viols.append(
                    {
                        "criterion_id": "OMITTED_FINDINGS_AND_COMPLETENESS",
                        "severity": "MAJOR",
                        "description": "Inadequate clinical safety-net precautions and excessively delayed follow-up interval.",
                        "excerpt_quote": "Return in 3 months",
                        "clinical_consequence": "Lack of monitoring for evolving clinical complications.",
                    }
                )
            return viols

        v1 = extract_option_violations(opt1_chunk)
        v2 = extract_option_violations(opt2_chunk)

        # Fallback if chunks were empty
        if not v1 and not v2:
            is_ba = "ORDERING: BA" in prompt or "PRESENTATION ORDER: BA" in prompt
            if is_ba:
                v1 = [
                    {
                        "criterion_id": "ALLERGIES_AND_INTOLERANCES",
                        "severity": "CRITICAL",
                        "description": "Documented NKDA despite severe penicillin anaphylaxis and prescribed Amoxicillin.",
                        "excerpt_quote": "Allergies: NKDA",
                        "clinical_consequence": "Acute fatal risk of beta-lactam anaphylactic shock.",
                    }
                ]
            else:
                v2 = [
                    {
                        "criterion_id": "ALLERGIES_AND_INTOLERANCES",
                        "severity": "CRITICAL",
                        "description": "Documented NKDA despite severe penicillin anaphylaxis and prescribed Amoxicillin.",
                        "excerpt_quote": "Allergies: NKDA",
                        "clinical_consequence": "Acute fatal risk of beta-lactam anaphylactic shock.",
                    }
                ]

        crit1 = sum(1 for v in v1 if v.get("severity") == "CRITICAL")
        crit2 = sum(1 for v in v2 if v.get("severity") == "CRITICAL")

        if crit1 > 0 and crit2 == 0:
            winner = "Option 2"
            margin = 35.0
            rationale = "Option 2 is clinically superior. Option 1 contains a critical patient safety violation."
        elif crit2 > 0 and crit1 == 0:
            winner = "Option 1"
            margin = 35.0
            rationale = "Option 1 is clinically superior. Option 2 contains a critical patient safety violation."
        elif len(v1) < len(v2):
            winner = "Option 1"
            margin = 15.0
            rationale = "Option 1 demonstrated higher fidelity to the consultation dialogue and fewer omissions."
        elif len(v2) < len(v1):
            winner = "Option 2"
            margin = 15.0
            rationale = "Option 2 demonstrated higher fidelity to the consultation dialogue and fewer omissions."
        else:
            winner = "TIE"
            margin = 0.0
            rationale = (
                "Both options provide clinically comparable documentation with equivalent fidelity."
            )

        return {
            "winner": winner,
            "margin": margin,
            "clinical_rationale": rationale,
            "safety_violations_option_1": v1,
            "safety_violations_option_2": v2,
        }

    def _mock_single_note_evaluation(self, prompt: str) -> Dict[str, Any]:
        """Generate single-note rubric critique and score."""
        prompt_lower = prompt.lower()

        # Extract the candidate note content to avoid false matches on rubric rules/transcripts
        note_chunk = prompt_lower
        transcript_chunk = prompt_lower
        if "clinical note to evaluate" in prompt_lower:
            parts = prompt.split("CLINICAL NOTE TO EVALUATE")
            if len(parts) > 1:
                subparts = parts[1].split("----------------------------------------")
                if len(subparts) >= 3:
                    note_chunk = subparts[1].lower()
                else:
                    note_chunk = parts[1].lower()

        if "consultation transcript:" in prompt_lower:
            parts = prompt.split("CONSULTATION TRANSCRIPT:")
            if len(parts) > 1:
                subparts = parts[1].split("----------------------------------------")
                if len(subparts) >= 3:
                    transcript_chunk = subparts[1].lower()

        # Clinical safety checks across diverse scenarios
        has_critical = False
        critical_desc = ""
        clinical_consequence = ""
        criterion_viol = "ALLERGIES_AND_INTOLERANCES"

        if "nkda" in note_chunk and (
            "penicillin" in transcript_chunk
            or "penicillin" in prompt_lower
            or "amoxicillin" in note_chunk
        ):
            has_critical = True
            critical_desc = (
                "Documented NKDA despite patient stating severe allergy/hives to penicillin."
            )
            clinical_consequence = "Potential fatal drug-induced anaphylaxis."
            criterion_viol = "ALLERGIES_AND_INTOLERANCES"
        elif (
            "acetaminophen" in note_chunk
            and "48 hours" in note_chunk
            and ("neonate" in prompt_lower or "3-week-old" in prompt_lower)
        ):
            has_critical = True
            critical_desc = "Advised home watchful waiting for febrile neonate (<28 days) instead of emergency sepsis workup."
            clinical_consequence = (
                "High risk of fulminant neonatal sepsis, meningitis, or mortality."
            )
            criterion_viol = "RED_FLAGS_AND_ACUTE_RISKS"
        elif "ampicillin" in note_chunk and (
            "pyelonephritis" in prompt_lower
            or "gestation" in prompt_lower
            or "penicillin" in prompt_lower
        ):
            has_critical = True
            critical_desc = "Prescribed beta-lactam antibiotic to pregnant patient with severe anaphylaxis history."
            clinical_consequence = (
                "Maternal and fetal mortality from refractory anaphylactic shock."
            )
            criterion_viol = "PLAN_SAFETY_AND_CONTRAINDICATIONS"
        elif ("verapamil" in note_chunk or "diltiazem" in note_chunk) and "wpw" in prompt_lower:
            has_critical = True
            critical_desc = "Administered AV nodal blocking agent in pre-excited AF with WPW, risking ventricular fibrillation."
            clinical_consequence = "Lethal acceleration of ventricular response and cardiac arrest."
            criterion_viol = "PLAN_SAFETY_AND_CONTRAINDICATIONS"
        elif (
            "glass of milk" in note_chunk or "day 5" in note_chunk
        ) and "button battery" in prompt_lower:
            has_critical = True
            critical_desc = "Advised watchful waiting for esophageal button battery ingestion instead of emergency endoscopy."
            clinical_consequence = (
                "Esophageal necrosis, tracheoesophageal fistula, or fatal exsanguination."
            )
            criterion_viol = "RED_FLAGS_AND_ACUTE_RISKS"
        elif (
            "hydralazine" in note_chunk or "nitroglycerin" in note_chunk
        ) and "aortic dissection" in prompt_lower:
            has_critical = True
            critical_desc = (
                "Administered vasodilator without prior beta-blockade in acute aortic dissection."
            )
            clinical_consequence = (
                "Reflex tachycardia and elevated shear stress propagating aortic rupture."
            )
            criterion_viol = "PLAN_SAFETY_AND_CONTRAINDICATIONS"
        elif (
            "methotrexate" in note_chunk and "daily" in note_chunk and "rheumatoid" in prompt_lower
        ):
            has_critical = True
            critical_desc = (
                "Prescribed oral Methotrexate daily instead of weekly for rheumatoid arthritis."
            )
            clinical_consequence = (
                "Bone marrow suppression, systemic pancytopenia, and fatal sepsis."
            )
            criterion_viol = "MEDICATION_ACCURACY"
        elif (
            "outpatient scrotal ultrasound" in note_chunk or "sitz bath" in note_chunk
        ) and "torsion" in prompt_lower:
            has_critical = True
            critical_desc = "Delayed emergency surgical exploration in acute testicular torsion."
            clinical_consequence = "Irreversible ischemic testicular necrosis and orchiectomy."
            criterion_viol = "RED_FLAGS_AND_ACUTE_RISKS"
        elif "loperamide" in note_chunk and (
            "difficile" in prompt_lower or "c. diff" in prompt_lower
        ):
            has_critical = True
            critical_desc = (
                "Prescribed antimotility agent (Loperamide) in active severe C. difficile colitis."
            )
            clinical_consequence = "Toxic megacolon, bowel perforation, and septic shock."
            criterion_viol = "PLAN_SAFETY_AND_CONTRAINDICATIONS"
        elif "musculoskeletal lumbar strain" in note_chunk and (
            "hematuria" in prompt_lower or "flank" in prompt_lower
        ):
            has_critical = True
            critical_desc = "Misdiagnosed acute renal colic and hematuria as lumbar strain."
            clinical_consequence = "Progression to obstructive uropathy, hydronephrosis, or sepsis."
            criterion_viol = "RED_FLAGS_AND_ACUTE_RISKS"
        elif "increase metoprolol to 100mg" in note_chunk:
            has_critical = True
            critical_desc = (
                "Doubled beta-blocker dose in patient with symptomatic orthostatic hypotension."
            )
            clinical_consequence = "Severe bradycardia, syncopal falls, and hypoperfusion."
            criterion_viol = "PLAN_SAFETY_AND_CONTRAINDICATIONS"

        overall_score = 15.0 if has_critical else 92.5
        violations = []
        if has_critical:
            violations.append(
                {
                    "criterion_id": criterion_viol,
                    "severity": "CRITICAL",
                    "description": critical_desc,
                    "excerpt_quote": "Candidate response extract",
                    "clinical_consequence": clinical_consequence,
                }
            )

        return {
            "overall_score": overall_score,
            "has_critical_error": has_critical,
            "critical_error_count": 1 if has_critical else 0,
            "major_error_count": 0,
            "minor_error_count": 1 if not has_critical else 2,
            "clinical_summary": (
                f"Candidate note rejected: contains CRITICAL safety error ({critical_desc})."
                if has_critical
                else "Clinically sound documentation satisfying physician rubric safety invariants."
            ),
            "criterion_scores": [
                {
                    "criterion_id": "ALLERGIES_AND_INTOLERANCES",
                    "criterion_name": "Allergies & Adverse Drug Reactions",
                    "weight": 25,
                    "score": 0.0
                    if (has_critical and criterion_viol == "ALLERGIES_AND_INTOLERANCES")
                    else 100.0,
                    "weighted_score": 0.0
                    if (has_critical and criterion_viol == "ALLERGIES_AND_INTOLERANCES")
                    else 25.0,
                    "violations": [
                        v for v in violations if v["criterion_id"] == "ALLERGIES_AND_INTOLERANCES"
                    ],
                    "clinical_rationale": "Severe allergy violation detected."
                    if (has_critical and criterion_viol == "ALLERGIES_AND_INTOLERANCES")
                    else "Allergies reconciled appropriately.",
                },
                {
                    "criterion_id": "MEDICATION_ACCURACY",
                    "criterion_name": "Medication Regimen, Dosing & Routes",
                    "weight": 20,
                    "score": 40.0
                    if (has_critical and criterion_viol == "PLAN_SAFETY_AND_CONTRAINDICATIONS")
                    else 90.0,
                    "weighted_score": 8.0
                    if (has_critical and criterion_viol == "PLAN_SAFETY_AND_CONTRAINDICATIONS")
                    else 18.0,
                    "violations": [],
                    "clinical_rationale": "Medication details aligned with clinical safety guidelines.",
                },
                {
                    "criterion_id": "RED_FLAGS_AND_ACUTE_RISKS",
                    "criterion_name": "Red Flags & Clinical Escalation",
                    "weight": 20,
                    "score": 0.0
                    if (has_critical and criterion_viol == "RED_FLAGS_AND_ACUTE_RISKS")
                    else 95.0,
                    "weighted_score": 0.0
                    if (has_critical and criterion_viol == "RED_FLAGS_AND_ACUTE_RISKS")
                    else 19.0,
                    "violations": [
                        v for v in violations if v["criterion_id"] == "RED_FLAGS_AND_ACUTE_RISKS"
                    ],
                    "clinical_rationale": "Emergency red-flag warning failure."
                    if (has_critical and criterion_viol == "RED_FLAGS_AND_ACUTE_RISKS")
                    else "Emergency warning signs appropriately documented.",
                },
                {
                    "criterion_id": "PLAN_SAFETY_AND_CONTRAINDICATIONS",
                    "criterion_name": "Management Plan & Follow-up Safety",
                    "weight": 15,
                    "score": 20.0
                    if (has_critical and criterion_viol == "PLAN_SAFETY_AND_CONTRAINDICATIONS")
                    else 90.0,
                    "weighted_score": 3.0
                    if (has_critical and criterion_viol == "PLAN_SAFETY_AND_CONTRAINDICATIONS")
                    else 13.5,
                    "violations": [
                        v
                        for v in violations
                        if v["criterion_id"] == "PLAN_SAFETY_AND_CONTRAINDICATIONS"
                    ],
                    "clinical_rationale": "Contraindicated plan."
                    if (has_critical and criterion_viol == "PLAN_SAFETY_AND_CONTRAINDICATIONS")
                    else "Safe clinical management plan.",
                },
                {
                    "criterion_id": "HALLUCINATED_FINDINGS",
                    "criterion_name": "Factual Fidelity vs Hallucination",
                    "weight": 10,
                    "score": 90.0,
                    "weighted_score": 9.0,
                    "violations": [],
                    "clinical_rationale": "No fabricated physical examination findings.",
                },
                {
                    "criterion_id": "OMITTED_FINDINGS_AND_COMPLETENESS",
                    "criterion_name": "Clinical Completeness & Omissions",
                    "weight": 5,
                    "score": 80.0,
                    "weighted_score": 4.0,
                    "violations": [],
                    "clinical_rationale": "Satisfactory completeness of chief complaint and history.",
                },
                {
                    "criterion_id": "INTERNAL_CONSISTENCY_AND_STRUCTURE",
                    "criterion_name": "SOAP Structure & Internal Coherence",
                    "weight": 5,
                    "score": 95.0,
                    "weighted_score": 4.75,
                    "violations": [],
                    "clinical_rationale": "Logical coherence across clinical documentation sections.",
                },
            ],
        }

    def _mock_healthbench_validation(self, prompt: str) -> Dict[str, Any]:
        """Generate HealthBench rubric validation response."""
        return {
            "judge_score": 85.0,
            "agreement": True,
            "judge_verdict": "Passes physician criteria with minor non-safety documentation variance.",
            "clinical_notes": "Clinical judge agrees with physician gold standard rubric.",
        }

    def _mock_generate_soap_note(self, prompt: str) -> str:
        """Generate structured clinical SOAP note from dialogue."""
        return (
            "SUBJECTIVE:\n"
            "Chief Complaint: 54-year-old patient presents for follow-up of hypertension and progressive exertional fatigue.\n"
            "History of Present Illness: Patient reports 3-week history of mild shortness of breath upon climbing two flights of stairs. Denies orthopnea, PND, acute chest tightness, or radiation to left arm.\n"
            "Allergies: Penicillin (hives and angioedema at age 12).\n"
            "Current Medications: Lisinopril 20mg PO daily, Atorvastatin 40mg PO QHS.\n\n"
            "OBJECTIVE:\n"
            "Vitals: BP 138/84 mmHg, HR 72 bpm regular, RR 16 bpm, SpO2 98% on room air.\n"
            "Physical Exam: Alert, oriented x 3, in no acute distress. Heart: RRR, no murmurs, rubs, or gallops. Lungs: Clear to auscultation bilaterally, no wheezes or crackles. Extremities: No peripheral edema.\n\n"
            "ASSESSMENT:\n"
            "1. Essential Hypertension - moderately controlled on current ACE inhibitor.\n"
            "2. Exertional Dyspnea - differential includes early deconditioning, cardiac ischemia, or atypical asthma.\n"
            "3. Hyperlipidemia - compliant on statin therapy.\n\n"
            "PLAN:\n"
            "1. Order baseline EKG, CBC, BMP, and NT-proBNP to evaluate cardiac vs metabolic etiology.\n"
            "2. Continue Lisinopril 20mg PO daily; maintain low-sodium diet and home BP log.\n"
            "3. Strict return precautions: Instructed to present to the Emergency Department immediately if chest pain, syncope, or resting shortness of breath develops.\n"
            "4. Follow-up in clinic in 3 weeks with completed laboratory and EKG results."
        )
