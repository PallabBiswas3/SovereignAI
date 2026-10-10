from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from controlplane.audit import JsonlFeedbackStore
from controlplane.checker import ControlPlane
from controlplane.schema import CheckReport, FeedbackEvent, Interaction, PromptCheckRequest
from controlplane.verification import LazyTransformersNLIScorer, build_adaptive_verification_from_env


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    service = control_plane.verification_service
    if service is not None and service.phase6 is not None:
        scorer = service.phase6.nli
        if isinstance(scorer, LazyTransformersNLIScorer):
            scorer.preload()
    app.state.ready = True
    try:
        yield
    finally:
        app.state.ready = False


app = FastAPI(
    title="ControlPlane.ai",
    version="0.2.0",
    description="Policy-driven safety gateway for enterprise generative AI.",
    lifespan=lifespan,
)
app.state.ready = False
control_plane = ControlPlane(verification_service=build_adaptive_verification_from_env())
feedback_store = JsonlFeedbackStore()


@app.get("/health")
def health() -> dict[str, str]:
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="ControlPlane verifier is not ready")
    return {"status": "ok"}


@app.get("/v1/policies")
def policies() -> dict[str, list[str]]:
    return {"profiles": control_plane.policy_repository.available()}


@app.post("/v1/check", response_model=CheckReport)
def check(interaction: Interaction) -> CheckReport:
    try:
        return control_plane.check(interaction)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/precheck", response_model=CheckReport)
def precheck(request: PromptCheckRequest) -> CheckReport:
    """Run input-side controls before sending a prompt to a model provider."""
    try:
        return control_plane.check(request.to_interaction())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/feedback")
def feedback(event: FeedbackEvent) -> dict[str, str]:
    return {"feedback_id": feedback_store.write(event)}
