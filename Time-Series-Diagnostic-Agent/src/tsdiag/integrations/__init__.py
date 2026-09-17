"""Stable integration boundaries for external tool and service hosts."""

from .diagnostic_tool import DIAGNOSE_TOOL_MANIFEST, diagnose_tool
from .claim_grounding import (
    ClaimReference,
    GroundingReport,
    SentenceClaims,
    verify_summary_claims,
)
from .presentation import (
    PresentationPolicy,
    PresentationPolicyRegistry,
    presentation_policy_registry,
    shape_diagnostic_output,
)

__all__ = [
    "DIAGNOSE_TOOL_MANIFEST",
    "diagnose_tool",
    "ClaimReference",
    "GroundingReport",
    "SentenceClaims",
    "verify_summary_claims",
    "PresentationPolicy",
    "PresentationPolicyRegistry",
    "presentation_policy_registry",
    "shape_diagnostic_output",
]
