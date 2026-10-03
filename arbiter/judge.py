"""Pairwise LLM clinical judge with position-bias swap mitigation and structured arbitration."""

from typing import Any, Dict, List, Optional, Tuple

from arbiter.llm.client import UnifiedLLMClient, get_llm_client
from arbiter.models import (
    CandidateNote,
    FinalVerdictEnum,
    OrderEvaluation,
    OrderVerdict,
    PairwiseVerdict,
    SafetyViolation,
    SeverityEnum,
)
from rubric.rubric_loader import ClinicalRubric, load_clinical_rubric

JUDGE_SYSTEM_PROMPT = """You are an expert physician arbitrator and medical peer reviewer.
Two different AI scribes have generated clinical SOAP notes for the SAME doctor-patient encounter.
Your objective is to judge which clinical note is clinically SAFER, MORE ACCURATE, and MORE FAITHFUL to the consultation dialogue.

You must rigorously apply the physician evaluation rubric:
1. Patient Safety & Allergies: Missing a documented allergy or reaction is an intolerable safety hazard.
2. Pharmacotherapy & Medications: Drug name, dose, frequency, and route must be completely accurate.
3. Red Flags & Urgent Risks: Any acute warning sign must be captured with appropriate escalation instructions.
4. Hallucinations vs Omissions: Do NOT reward notes that invent physical exam maneuvers or symptoms not present in the dialogue.
5. Critical Safety Override: If one note has a CRITICAL safety error and the other note does not, the note WITHOUT the critical error MUST win.

Return your evaluation strictly in valid JSON matching this schema:
{
  "winner": "Option 1" | "Option 2" | "TIE",
  "margin": <float 0.0 - 100.0>,
  "clinical_rationale": "<comprehensive physician clinical argument comparing both options>",
  "safety_violations_option_1": [
    {
      "criterion_id": "<CRITERION_ID>",
      "severity": "CRITICAL" | "MAJOR" | "MINOR" | "BENIGN",
      "description": "<error description>",
      "excerpt_quote": "<quote from note or transcript>",
      "clinical_consequence": "<potential patient harm>"
    }
  ],
  "safety_violations_option_2": [
    {
      "criterion_id": "<CRITERION_ID>",
      "severity": "CRITICAL" | "MAJOR" | "MINOR" | "BENIGN",
      "description": "<error description>",
      "excerpt_quote": "<quote from note or transcript>",
      "clinical_consequence": "<potential patient harm>"
    }
  ]
}
"""


class ClinicalJudge:
    """Pairwise clinical arbiter with position-bias detection and mitigation."""

    def __init__(
        self,
        model_name: str = "mock-judge",
        rubric: Optional[ClinicalRubric] = None,
        client: Optional[UnifiedLLMClient] = None,
    ):
        self.model_name = model_name
        self.rubric = rubric or load_clinical_rubric()
        self.client = client or get_llm_client(model_name)

    def _parse_violations(self, raw_list: List[Dict[str, Any]]) -> List[SafetyViolation]:
        violations = []
        for item in raw_list:
            sev_str = str(item.get("severity", "MINOR")).upper()
            if sev_str not in SeverityEnum.__members__:
                sev_str = "MINOR"
            violations.append(
                SafetyViolation(
                    criterion_id=str(item.get("criterion_id", "UNKNOWN")),
                    severity=SeverityEnum(sev_str),
                    description=str(item.get("description", "")),
                    excerpt_quote=item.get("excerpt_quote"),
                    clinical_consequence=str(item.get("clinical_consequence", "")),
                )
            )
        return violations

    def _evaluate_ordering(
        self,
        transcript: str,
        opt1_text: str,
        opt2_text: str,
        order_tag: str,  # "AB" or "BA"
    ) -> Tuple[OrderEvaluation, List[SafetyViolation], List[SafetyViolation]]:
        """Run a single pairwise comparison ordering."""
        prompt = f"""CONSULTATION TRANSCRIPT:
----------------------------------------
{transcript.strip()}
----------------------------------------

OPTION 1 CLINICAL NOTE:
----------------------------------------
{opt1_text.strip()}
----------------------------------------

OPTION 2 CLINICAL NOTE:
----------------------------------------
{opt2_text.strip()}
----------------------------------------

PRESENTATION ORDER: {order_tag}

{self.rubric.get_prompt_text()}

Compare Option 1 and Option 2 based on the transcript and rubric.
Output ONLY the requested JSON object.
"""
        response = self.client.generate(
            prompt=prompt,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            json_mode=True,
        )

        parsed = response.parsed_json or {}
        winner_raw = str(parsed.get("winner", "TIE")).strip().upper()
        if "OPTION 1" in winner_raw or winner_raw == "1" or winner_raw == "OPTION1":
            winner_in_order = OrderVerdict.A if order_tag == "AB" else OrderVerdict.B
        elif "OPTION 2" in winner_raw or winner_raw == "2" or winner_raw == "OPTION2":
            winner_in_order = OrderVerdict.B if order_tag == "AB" else OrderVerdict.A
        elif "NOTE A" in winner_raw or "OPTION A" in winner_raw or winner_raw == "A":
            winner_in_order = OrderVerdict.A
        elif "NOTE B" in winner_raw or "OPTION B" in winner_raw or winner_raw == "B":
            winner_in_order = OrderVerdict.B
        else:
            winner_in_order = OrderVerdict.TIE

        margin = float(parsed.get("margin", 0.0))
        rationale = str(parsed.get("clinical_rationale", "No rationale provided."))

        raw_opt1_viols = parsed.get("safety_violations_option_1", [])
        raw_opt2_viols = parsed.get("safety_violations_option_2", [])

        opt1_viols = self._parse_violations(raw_opt1_viols)
        opt2_viols = self._parse_violations(raw_opt2_viols)

        # Map violations back to Note A and Note B
        if order_tag == "AB":
            note_a_viols = opt1_viols
            note_b_viols = opt2_viols
        else:
            note_a_viols = opt2_viols
            note_b_viols = opt1_viols

        order_eval = OrderEvaluation(
            order=order_tag,
            winner=winner_in_order,
            margin=margin,
            rationale=rationale,
            note_a_violations=note_a_viols,
            note_b_violations=note_b_viols,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=response.cost_usd,
        )
        return order_eval, note_a_viols, note_b_viols

    def arbitrate(
        self,
        transcript: str,
        note_a: CandidateNote | str,
        note_b: CandidateNote | str,
        encounter_id: Optional[str] = None,
    ) -> PairwiseVerdict:
        """
        Conduct pairwise clinical arbitration with dual-presentation swap mitigation.
        Step 1: Present (Note A, Note B).
        Step 2: Present (Note B, Note A).
        Step 3: Combine verdicts, detect position bias, and apply clinical safety override.
        """
        text_a = note_a.raw_text if isinstance(note_a, CandidateNote) else str(note_a)
        text_b = note_b.raw_text if isinstance(note_b, CandidateNote) else str(note_b)
        model_a = note_a.model_name if isinstance(note_a, CandidateNote) else "Model A"
        model_b = note_b.model_name if isinstance(note_b, CandidateNote) else "Model B"

        # 1. Run AB ordering
        eval_ab, ab_viols_a, ab_viols_b = self._evaluate_ordering(
            transcript=transcript,
            opt1_text=text_a,
            opt2_text=text_b,
            order_tag="AB",
        )

        # 2. Run BA ordering (swap mitigation)
        eval_ba, ba_viols_a, ba_viols_b = self._evaluate_ordering(
            transcript=transcript,
            opt1_text=text_b,
            opt2_text=text_a,
            order_tag="BA",
        )

        # Consolidated violation counts
        all_viols_a = ab_viols_a + [
            v for v in ba_viols_a if v.description not in [x.description for x in ab_viols_a]
        ]
        all_viols_b = ab_viols_b + [
            v for v in ba_viols_b if v.description not in [x.description for x in ab_viols_b]
        ]

        crit_a = sum(1 for v in all_viols_a if v.severity == SeverityEnum.CRITICAL)
        crit_b = sum(1 for v in all_viols_b if v.severity == SeverityEnum.CRITICAL)

        # Arbitration logic: Check presentation consistency first (order invariance invariant)
        raw_orderings_disagree = eval_ab.winner != eval_ba.winner
        position_bias_detected = raw_orderings_disagree
        final_verdict = FinalVerdictEnum.TIE
        winning_note_id = None
        winning_model = None
        confidence = 0.5

        if raw_orderings_disagree:
            # Order inconsistency detected (position bias): inhibit autonomous sign-off and route to physician
            position_bias_detected = True
            final_verdict = FinalVerdictEnum.INCONCLUSIVE_POSITION_BIAS
            winning_note_id = None
            winning_model = None
            confidence = 0.40
            clinical_summary = (
                f"Position bias detected: Presentation AB selected '{eval_ab.winner.value}' "
                f"while Presentation BA selected '{eval_ba.winner.value}'. "
                f"Verdict is deemed INCONCLUSIVE pending mandatory human physician review."
            )
        elif crit_a > 0 and crit_b == 0:
            final_verdict = FinalVerdictEnum.NOTE_B_WINS
            winning_note_id = "Note B"
            winning_model = model_b
            confidence = 0.98
            clinical_summary = (
                "Note B decisively won due to CRITICAL safety violation in Note A."
            )
        elif crit_b > 0 and crit_a == 0:
            final_verdict = FinalVerdictEnum.NOTE_A_WINS
            winning_note_id = "Note A"
            winning_model = model_a
            confidence = 0.98
            clinical_summary = (
                "Note A decisively won due to CRITICAL safety violation in Note B."
            )
        elif eval_ab.winner == OrderVerdict.A:
            final_verdict = FinalVerdictEnum.NOTE_A_WINS
            winning_note_id = "Note A"
            winning_model = model_a
            confidence = round(0.70 + (eval_ab.margin + eval_ba.margin) / 400.0, 2)
            confidence = min(0.99, max(0.60, confidence))
            clinical_summary = (
                "Note A consistently won across both forward and swapped orderings."
            )
        elif eval_ab.winner == OrderVerdict.B:
            final_verdict = FinalVerdictEnum.NOTE_B_WINS
            winning_note_id = "Note B"
            winning_model = model_b
            confidence = round(0.70 + (eval_ab.margin + eval_ba.margin) / 400.0, 2)
            confidence = min(0.99, max(0.60, confidence))
            clinical_summary = (
                "Note B consistently won across both forward and swapped orderings."
            )
        else:
            final_verdict = FinalVerdictEnum.TIE
            confidence = 0.50
            clinical_summary = (
                "Both presentations yielded an indistinguishable clinical quality tie."
            )

        combined_rationale = (
            f"CLINICAL ARBITRATION SUMMARY: {clinical_summary}\n\n"
            f"[Ordering AB Rationale]: {eval_ab.rationale}\n\n"
            f"[Ordering BA Rationale]: {eval_ba.rationale}"
        )

        safety_summary = (
            f"Safety Comparison: Note A had {crit_a} critical and {len(all_viols_a) - crit_a} non-critical violations; "
            f"Note B had {crit_b} critical and {len(all_viols_b) - crit_b} non-critical violations."
        )

        total_prompt_tokens = eval_ab.prompt_tokens + eval_ba.prompt_tokens
        total_completion_tokens = eval_ab.completion_tokens + eval_ba.completion_tokens
        total_cost_usd = round(eval_ab.cost_usd + eval_ba.cost_usd, 6)

        return PairwiseVerdict(
            encounter_id=encounter_id,
            final_verdict=final_verdict,
            winning_note_id=winning_note_id,
            winning_model=winning_model,
            confidence=confidence,
            position_bias_detected=position_bias_detected,
            eval_ab=eval_ab,
            eval_ba=eval_ba,
            clinical_rationale=combined_rationale,
            safety_comparison_summary=safety_summary,
            note_a_critical_count=crit_a,
            note_b_critical_count=crit_b,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_cost_usd=total_cost_usd,
            judge_model=self.model_name,
        )
