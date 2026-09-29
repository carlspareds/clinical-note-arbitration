"""Loader and schema validation for clinical rubrics."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class SeverityLevel(BaseModel):
    level: int
    name: str
    penalty_points: int
    description: str
    examples: List[str] = Field(default_factory=list)


class Criterion(BaseModel):
    id: str
    name: str
    category: str
    weight: int
    description: str
    clinical_rationale: str
    rubric_rules: List[str] = Field(default_factory=list)


class ClinicalRubric(BaseModel):
    rubric_metadata: Dict[str, Any]
    severity_levels: Dict[str, SeverityLevel]
    criteria: List[Criterion]
    arbitration_decision_rules: Dict[str, Any]

    def get_criterion(self, criterion_id: str) -> Optional[Criterion]:
        for c in self.criteria:
            if c.id.upper() == criterion_id.upper():
                return c
        return None

    def get_prompt_text(self) -> str:
        """Render rubric as formatted string for LLM prompts."""
        lines = [
            f"# {self.rubric_metadata.get('name', 'Clinical Rubric')}",
            f"{self.rubric_metadata.get('description', '')}\n",
            "## SEVERITY CLASSIFICATION:",
        ]
        for key, sev in self.severity_levels.items():
            lines.append(f"- **{key} ({sev.name})**: {sev.description}")
            if sev.examples:
                lines.append(f"  Examples: {'; '.join(sev.examples[:3])}")

        lines.append("\n## CLINICALLY WEIGHTED CRITERIA:")
        for c in self.criteria:
            lines.append(
                f"### {c.name} [ID: {c.id}] (Weight: {c.weight}/100, Category: {c.category})"
            )
            lines.append(f"- Description: {c.description}")
            lines.append(f"- Clinical Risk: {c.clinical_rationale}")
            if c.rubric_rules:
                lines.append("  Rules:")
                for rule in c.rubric_rules:
                    lines.append(f"    * {rule}")

        lines.append("\n## ARBITRATION DECISION PROTOCOL:")
        for name, rule in self.arbitration_decision_rules.items():
            lines.append(f"- **{name}**: {rule.get('rule', '')}")

        return "\n".join(lines)


def load_clinical_rubric(rubric_path: Optional[Path] = None) -> ClinicalRubric:
    """Load rubric YAML from default or custom path."""
    if rubric_path is None:
        rubric_path = Path(__file__).resolve().parent / "clinical_rubric.yaml"

    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric file not found at {rubric_path}")

    with open(rubric_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ClinicalRubric.model_validate(data)
