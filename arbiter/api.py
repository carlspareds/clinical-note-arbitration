"""FastAPI application providing clinical note arbitration, scoring, and web dashboard endpoints."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from arbiter.generator import NoteGenerator
from arbiter.judge import ClinicalJudge
from arbiter.models import PairwiseVerdict, SingleNoteEvaluation
from arbiter.scorer import SingleNoteScorer

app = FastAPI(
    title="Clinical Note Arbitration API",
    description="LLM-as-a-Judge for Clinical Documentation with Physician Rubric and Position-Bias Mitigation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent / "web"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class ArbitrateRequest(BaseModel):
    transcript: str = Field(..., description="Doctor-patient dialogue transcript")
    note_a: str = Field(..., description="Candidate clinical SOAP Note A")
    note_b: str = Field(..., description="Candidate clinical SOAP Note B")
    judge_model: Optional[str] = Field("mock-judge", description="LLM judge model identifier")
    encounter_id: Optional[str] = Field(None, description="Optional encounter identifier")


class GenerateRequest(BaseModel):
    transcript: str = Field(..., description="Doctor-patient dialogue transcript")
    model_a: Optional[str] = Field("mock-generator", description="Model for Note A")
    model_b: Optional[str] = Field("mock-generator", description="Model for Note B")


class ScoreRequest(BaseModel):
    transcript: str = Field(..., description="Doctor-patient dialogue transcript")
    note: str = Field(..., description="Candidate clinical SOAP note")
    judge_model: Optional[str] = Field("mock-judge", description="LLM judge model identifier")
    note_id: Optional[str] = Field("Note A", description="Identifier for candidate note")


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the interactive side-by-side arbitration viewer."""
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found")
    return index_file.read_text(encoding="utf-8")


@app.get("/api/health")
def health_check():
    """Health status and service metadata."""
    return {
        "status": "healthy",
        "service": "clinical-note-arbitration",
        "version": "0.1.0",
        "supported_providers": [
            "openai",
            "anthropic",
            "gemini",
            "openrouter",
            "gradient",
            "ollama",
            "mock",
        ],
    }


@app.get("/api/samples")
def get_sample_encounters() -> List[Dict[str, Any]]:
    """Return pre-packaged de-identified clinical encounters for interactive testing."""
    sample_file = DATA_DIR / "samples" / "aci_bench_samples.json"
    if sample_file.exists():
        with open(sample_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.post("/api/arbitrate", response_model=PairwiseVerdict)
def arbitrate_notes(req: ArbitrateRequest):
    """Run dual-presentation pairwise arbitration with position-bias check."""
    judge = ClinicalJudge(model_name=req.judge_model or "mock-judge")
    verdict = judge.arbitrate(
        transcript=req.transcript,
        note_a=req.note_a,
        note_b=req.note_b,
        encounter_id=req.encounter_id or "API-ENCOUNTER",
    )
    return verdict


@app.post("/api/score", response_model=SingleNoteEvaluation)
def score_single_note(req: ScoreRequest):
    """Evaluate a single clinical note against the physician rubric."""
    scorer = SingleNoteScorer(model_name=req.judge_model or "mock-judge")
    evaluation = scorer.evaluate_note(
        transcript=req.transcript,
        note=req.note,
        note_id=req.note_id or "Note A",
    )
    return evaluation


@app.post("/api/generate")
def generate_competing_notes(req: GenerateRequest):
    """Generate two competing SOAP notes from the same transcript."""
    gen_a = NoteGenerator(model_name=req.model_a or "mock-generator")
    gen_b = NoteGenerator(model_name=req.model_b or "mock-generator")

    note_a = gen_a.generate_note(req.transcript, note_id="Note A")
    note_b = gen_b.generate_note(req.transcript, note_id="Note B")

    return {
        "note_a": note_a.model_dump(),
        "note_b": note_b.model_dump(),
    }
