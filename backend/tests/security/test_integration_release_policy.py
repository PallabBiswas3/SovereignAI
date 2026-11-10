"""The server, not the caller, decides how strictly integrated analysis is released.

Before 10 November 2026 a caller could send policy_profile=customer_support and
consequential=false (skipping NLI for its own answer), or a candidate_response
that was then written into a "Verified recommendation" artifact.
"""
import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api import integrations as api
from app.core.database import Base
from app.identity.models import Principal
from app.integrations.models import (
    CONTROLPLANE_DEFAULT_PROFILE,
    IntegratedAnalysisRequest,
    resolve_controlplane_profile,
)
from diagnostic_fixtures import payload
from tsdiag.input_contract import WORKFLOWS


@pytest.mark.parametrize("profile", ["customer_support", "unknown_profile"])
def test_caller_cannot_select_a_weaker_release_profile(profile):
    with pytest.raises(ValidationError):
        IntegratedAnalysisRequest(query="Assess Pump-102", policy_profile=profile)


def test_caller_cannot_mark_industrial_analysis_non_consequential():
    with pytest.raises(ValidationError):
        IntegratedAnalysisRequest(query="Assess Pump-102", consequential=False)


@pytest.mark.parametrize("profile", ["internal_assistant", "regulated_decision_support"])
def test_default_and_stricter_profiles_are_accepted(profile):
    request = IntegratedAnalysisRequest(query="Assess Pump-102", policy_profile=profile)
    assert request.policy_profile == profile and request.consequential is True


@pytest.mark.parametrize("use_case, expected", [
    ("internal_assistant", "internal_assistant"),
    ("regulated_decision_support", "regulated_decision_support"),
    ("customer_support", CONTROLPLANE_DEFAULT_PROFILE),
    ("engineering", CONTROLPLANE_DEFAULT_PROFILE),
    ("finance", CONTROLPLANE_DEFAULT_PROFILE),
    (None, CONTROLPLANE_DEFAULT_PROFILE),
])
def test_workbench_use_case_maps_only_to_permitted_profiles(use_case, expected):
    assert resolve_controlplane_profile(use_case) == expected


@pytest.fixture
def released_analysis(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    principal = Principal(user_id="user-a", email="synthetic@example.invalid", display_name="Synthetic",
                          organization_id="org-a", workspace_ids=["workspace-a"])
    materialized = []

    class Orchestrator:
        async def analyze(self, request, principal):
            return SimpleNamespace(released=True, status="released", service_status={}, run_id="run-synthetic-1",
                                   controlplane={"decision": {"action": "allow"}})

    monkeypatch.setattr(api, "IndustrialIntegrationOrchestrator", Orchestrator)
    monkeypatch.setattr(api, "_materialize_pump_102_outputs", lambda *args: materialized.append(args))
    with Session(engine) as db:
        yield principal, db, materialized
    engine.dispose()


def _analyze(setup, **kwargs):
    principal, db, _ = setup
    request = IntegratedAnalysisRequest(query="Assess Pump-102", diagnostic=payload(next(iter(WORKFLOWS))), **kwargs)
    return asyncio.run(api.integrated_analysis(request, principal, db))


def test_client_supplied_response_is_never_materialized_as_verified(released_analysis):
    _analyze(released_analysis, candidate_response="Pump-102 is healthy; no action is needed.")
    assert released_analysis[2] == []


def test_model_generated_released_response_is_still_materialized(released_analysis):
    _analyze(released_analysis)
    assert len(released_analysis[2]) == 1
