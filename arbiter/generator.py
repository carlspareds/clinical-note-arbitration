"""Clinical SOAP note generation from doctor-patient consultation dialogues."""

import re
from typing import Optional

from arbiter.llm.client import UnifiedLLMClient, get_llm_client
from arbiter.models import CandidateNote, SOAPSections

GENERATOR_SYSTEM_PROMPT = """You are an expert clinical medical scribe and documentation specialist.
Your task is to convert an ambient doctor-patient encounter audio transcript into a professional, clinically accurate SOAP note (Subjective, Objective, Assessment, Plan).

RULES:
1. ONLY include clinical information directly stated, affirmed, or measured in the transcript.
2. NEVER hallucinate physical examination findings, lab results, vital signs, or patient history not present in the dialogue.
3. Explicitly capture ALL stated drug allergies, adverse drug reactions, and pertinent negative histories.
4. Record exact medication names, doses, routes, and frequencies.
5. Emphasize acute red flags, return precautions, and explicit contingency plans in the Assessment and Plan.
6. Format output strictly with clear headers:
   SUBJECTIVE:
   OBJECTIVE:
   ASSESSMENT:
   PLAN:
"""


def parse_soap_sections(text: str) -> Optional[SOAPSections]:
    """Extract Subjective, Objective, Assessment, Plan sections from note text."""
    patterns = {
        "subjective": r"(?:^|\n)\s*(?:#+\s*)?(?:SUBJECTIVE|S)\s*:(.*?)(?=(?:^|\n)\s*(?:#+\s*)?(?:OBJECTIVE|O|ASSESSMENT|A|PLAN|P)\s*:|\Z)",
        "objective": r"(?:^|\n)\s*(?:#+\s*)?(?:OBJECTIVE|O)\s*:(.*?)(?=(?:^|\n)\s*(?:#+\s*)?(?:ASSESSMENT|A|PLAN|P)\s*:|\Z)",
        "assessment": r"(?:^|\n)\s*(?:#+\s*)?(?:ASSESSMENT|A)\s*:(.*?)(?=(?:^|\n)\s*(?:#+\s*)?(?:PLAN|P)\s*:|\Z)",
        "plan": r"(?:^|\n)\s*(?:#+\s*)?(?:PLAN|P)\s*:(.*?)$",
    }

    extracted = {}
    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted[section] = match.group(1).strip()
        else:
            extracted[section] = ""

    if any(extracted.values()):
        return SOAPSections(
            subjective=extracted.get("subjective", ""),
            objective=extracted.get("objective", ""),
            assessment=extracted.get("assessment", ""),
            plan=extracted.get("plan", ""),
        )
    return None


class NoteGenerator:
    """Generates structured SOAP notes from consultation dialogues."""

    def __init__(
        self, model_name: str = "mock-generator", client: Optional[UnifiedLLMClient] = None
    ):
        self.model_name = model_name
        self.client = client or get_llm_client(model_name)

    def generate_note(
        self,
        transcript: str,
        note_id: str = "Note",
        custom_instructions: Optional[str] = None,
        temperature: float = 0.1,
    ) -> CandidateNote:
        """Generate a candidate SOAP note from a doctor-patient dialogue transcript."""
        prompt = f"""CONSULTATION TRANSCRIPT:
----------------------------------------
{transcript.strip()}
----------------------------------------

Please generate the clinical SOAP note based strictly on the above transcript.
{f"ADDITIONAL CLINICAL INSTRUCTIONS: {custom_instructions}" if custom_instructions else ""}
"""
        response = self.client.generate(
            prompt=prompt,
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            temperature=temperature,
            json_mode=False,
        )

        note_text = response.content.strip()
        sections = parse_soap_sections(note_text)

        return CandidateNote(
            note_id=note_id,
            model_name=self.model_name,
            raw_text=note_text,
            sections=sections,
        )
