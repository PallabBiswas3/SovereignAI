from __future__ import annotations

from controlplane.policy.models import PolicyProfile
from controlplane.schema import EnforcementAction, Finding


def redact_response(response: str, findings: list[Finding]) -> str:
    replacements = []
    for finding in findings:
        if finding.span is None or finding.replacement is None:
            continue
        start, end = finding.span
        if 0 <= start < end <= len(response):
            replacements.append((start, end, finding.replacement))

    output = response
    last_start = len(response) + 1
    for start, end, replacement in sorted(replacements, reverse=True):
        if end > last_start:  # overlapping lower-priority finding
            continue
        output = output[:start] + replacement + output[end:]
        last_start = start
    return output


def transform_response(
    response: str,
    action: EnforcementAction,
    findings: list[Finding],
    profile: PolicyProfile,
) -> str:
    if action == EnforcementAction.BLOCK:
        return profile.blocked_response
    if action == EnforcementAction.REVIEW:
        return profile.review_response
    if action == EnforcementAction.REDACT:
        return redact_response(response, findings)
    if action == EnforcementAction.WARN:
        return f"{response}\n\n⚠ {profile.warning_template}"
    return response
