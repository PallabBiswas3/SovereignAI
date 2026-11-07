from __future__ import annotations

from collections.abc import Iterable

from controlplane.policy.models import PolicyProfile
from controlplane.schema import SEVERITY_RANK, EnforcementAction, Finding


def redact_response(response: str, findings: list[Finding]) -> str:
    spans: list[tuple[int, int, Finding]] = []
    for finding in findings:
        if finding.span is None or finding.replacement is None:
            continue
        start, end = finding.span
        if 0 <= start < end <= len(response):
            spans.append((start, end, finding))

    # Overlapping detections (for example an API key inside a bearer token)
    # are redacted as one union. Keeping only one span would leave the
    # non-overlapping fragment of the other credential in the output.
    groups: list[tuple[int, int, list[Finding]]] = []
    for start, end, finding in sorted(spans, key=lambda item: (item[0], item[1])):
        if groups and start < groups[-1][1]:
            group_start, group_end, members = groups[-1]
            groups[-1] = (group_start, max(group_end, end), [*members, finding])
        else:
            groups.append((start, end, [finding]))

    output = response
    for start, end, members in reversed(groups):
        primary = max(
            members,
            key=lambda item: (SEVERITY_RANK[item.severity], item.span[1] - item.span[0]),  # type: ignore[index]
        )
        output = output[:start] + str(primary.replacement) + output[end:]
    return output


def transform_response(
    response: str,
    action: EnforcementAction,
    findings: list[Finding],
    profile: PolicyProfile,
    matched_rules: Iterable[str] = (),
) -> str:
    if action == EnforcementAction.BLOCK:
        return profile.blocked_response
    if action == EnforcementAction.REVIEW:
        return profile.review_response
    if action == EnforcementAction.REDACT:
        redacted = redact_response(response, findings)
        # Redaction outranks a warning, but must not silently discard it: a
        # matched warning rule (e.g. a failed safety control) stays visible.
        matched = set(matched_rules)
        if any(rule.id in matched and rule.action == EnforcementAction.WARN for rule in profile.rules):
            return f"{redacted}\n\n⚠ {profile.warning_template}"
        return redacted
    if action == EnforcementAction.WARN:
        return f"{response}\n\n⚠ {profile.warning_template}"
    return response
