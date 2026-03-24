"""FastAPI server wrapping the guardrails library."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from guardrails import GuardrailsEngine
from guardrails.types import Action, RedactionStrategy, Severity

logger = structlog.get_logger()

# Global engine instance
engine: GuardrailsEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize guardrails engine on startup."""
    global engine
    policy_dir = Path(__file__).parent.parent.parent / "policies"
    engine = GuardrailsEngine()
    if policy_dir.exists():
        engine.load_policies_dir(policy_dir)
        logger.info("Loaded policies", directory=str(policy_dir))
    yield
    engine = None


app = FastAPI(
    title="AI Guardrails API",
    description="Production-grade AI safety and content guardrails",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Request/Response Models ---

class ScanRequest(BaseModel):
    """Request to scan text against active policies."""
    text: str = Field(..., min_length=1, max_length=100_000, description="Text to scan")
    policy_names: list[str] | None = Field(None, description="Specific policies to apply")


class ScanResponse(BaseModel):
    """Scan result response."""
    is_safe: bool
    action: str
    detections: list[dict[str, Any]]
    policy_violations: list[dict[str, Any]]
    metadata: dict[str, Any]


class RedactRequest(BaseModel):
    """Request to redact PII from text."""
    text: str = Field(..., min_length=1, max_length=100_000)
    strategy: str = Field("replace", description="Redaction strategy: mask, hash, replace, remove")
    entity_types: list[str] | None = Field(None, description="Specific entity types to redact")


class RedactResponse(BaseModel):
    """Redaction result response."""
    original_text: str
    redacted_text: str
    redactions: list[dict[str, Any]]


class ValidateRequest(BaseModel):
    """Request to validate prompt against injection rules."""
    text: str = Field(..., min_length=1, max_length=100_000)
    sensitivity: str = Field("medium", description="Sensitivity level: low, medium, high, paranoid")


class ValidateResponse(BaseModel):
    """Validation result response."""
    is_safe: bool
    is_injection: bool
    confidence: float
    matched_patterns: list[dict[str, Any]]
    severity: str


class PolicyResponse(BaseModel):
    """Policy information response."""
    name: str
    version: str
    description: str
    rule_count: int
    rules: list[dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    policies_loaded: int


class MetricsResponse(BaseModel):
    """Metrics response."""
    uptime_seconds: float
    requests_total: int
    scans_total: int
    detections_total: int


# --- Metrics tracking ---
_start_time = time.time()
_metrics = {"requests_total": 0, "scans_total": 0, "detections_total": 0}


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    policy_count = len(engine.policies) if engine else 0
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        policies_loaded=policy_count,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """Metrics endpoint."""
    return MetricsResponse(
        uptime_seconds=time.time() - _start_time,
        **_metrics,
    )


@app.post("/api/v1/scan", response_model=ScanResponse)
async def scan_text(request: ScanRequest) -> ScanResponse:
    """Scan text against all active policies."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    _metrics["requests_total"] += 1
    _metrics["scans_total"] += 1

    result = await engine.scan(request.text)
    _metrics["detections_total"] += len(result.detections)

    return ScanResponse(
        is_safe=result.is_safe,
        action=result.action.value,
        detections=[
            {
                "entity_type": d.entity_type,
                "text": d.text,
                "start": d.start,
                "end": d.end,
                "confidence": d.confidence,
                "detector": d.detector,
                "severity": d.severity.value,
            }
            for d in result.detections
        ],
        policy_violations=[
            {
                "rule_name": v.rule_name,
                "policy_name": v.policy_name,
                "severity": v.severity.value,
                "action": v.action.value,
                "message": v.message,
            }
            for v in result.policy_violations
        ],
        metadata=result.metadata,
    )


@app.post("/api/v1/redact", response_model=RedactResponse)
async def redact_text(request: RedactRequest) -> RedactResponse:
    """Redact PII from text."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    _metrics["requests_total"] += 1

    try:
        strategy = RedactionStrategy(request.strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy: {request.strategy}. Must be one of: mask, hash, replace, remove",
        )

    result = await engine.redact(request.text, strategy=strategy)

    return RedactResponse(
        original_text=result.original_text,
        redacted_text=result.redacted_text,
        redactions=[
            {
                "entity_type": r.entity_type,
                "original": r.original,
                "replacement": r.replacement,
                "start": r.start,
                "end": r.end,
                "strategy": r.strategy.value,
            }
            for r in result.redactions
        ],
    )


@app.post("/api/v1/validate", response_model=ValidateResponse)
async def validate_prompt(request: ValidateRequest) -> ValidateResponse:
    """Validate a prompt against injection detection rules."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    _metrics["requests_total"] += 1

    result = await engine.validate_prompt(request.text)

    return ValidateResponse(
        is_safe=not result.is_injection,
        is_injection=result.is_injection,
        confidence=result.confidence,
        matched_patterns=[
            {
                "name": p.pattern_name,
                "category": p.category.value,
                "severity": p.severity.value,
                "matched_text": p.matched_text,
            }
            for p in result.matched_patterns
        ],
        severity=result.severity.value,
    )


@app.get("/api/v1/policies", response_model=list[PolicyResponse])
async def list_policies() -> list[PolicyResponse]:
    """List all active policies."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    return [
        PolicyResponse(
            name=p.name,
            version=p.version,
            description=p.description,
            rule_count=len(p.rules),
            rules=[
                {
                    "name": r.name,
                    "detector_type": r.detector_type,
                    "action": r.action.value,
                    "severity": r.severity.value,
                    "enabled": r.enabled,
                }
                for r in p.rules
            ],
        )
        for p in engine.policies
    ]
