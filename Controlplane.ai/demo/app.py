from __future__ import annotations

import html
import os

import streamlit as st

from controlplane import ControlPlane, Interaction
from controlplane.schema import EnforcementAction
from controlplane.verification import build_adaptive_verification_from_env


st.set_page_config(
    page_title="ControlPlane.ai",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --cp-navy: #071426;
        --cp-panel: #0e2037;
        --cp-cyan: #38bdf8;
        --cp-green: #34d399;
        --cp-amber: #fbbf24;
        --cp-red: #fb7185;
        --cp-muted: #94a3b8;
        --cp-border: rgba(148, 163, 184, 0.22);
    }
    .stApp {
        background: radial-gradient(circle at 85% 8%, rgba(14,165,233,.10), transparent 25rem),
                    linear-gradient(180deg, #07111f 0%, var(--cp-navy) 100%);
    }
    [data-testid="stSidebar"] {
        background: rgba(7, 20, 38, .96);
        border-right: 1px solid var(--cp-border);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 4rem; }
    .cp-hero {
        padding: 1.4rem 1.55rem; margin-bottom: 1.25rem;
        border: 1px solid var(--cp-border); border-radius: 18px;
        background: linear-gradient(120deg, rgba(14,32,55,.94), rgba(8,47,73,.78));
        box-shadow: 0 18px 60px rgba(0,0,0,.18);
    }
    .cp-eyebrow { color: var(--cp-cyan); font-size: .76rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; }
    .cp-hero h1 { margin: .25rem 0 .2rem; font-size: 2.05rem; letter-spacing: -.035em; }
    .cp-hero p { margin: 0; color: #b8c6d9; max-width: 850px; }
    .cp-decision {
        padding: 1rem 1.15rem; margin: .35rem 0 1rem; border-radius: 14px;
        border-left: 5px solid var(--decision-color);
        background: rgba(14, 32, 55, .90);
    }
    .cp-decision-title { font-size: 1.05rem; font-weight: 750; color: var(--decision-color); }
    .cp-decision-reason { color: #cbd5e1; margin-top: .25rem; font-size: .9rem; }
    .cp-status {
        display: inline-flex; align-items: center; gap: .45rem; padding: .42rem .65rem;
        border: 1px solid var(--cp-border); border-radius: 999px; color: #dbeafe;
        background: rgba(14,32,55,.75); font-size: .78rem; font-weight: 650;
    }
    .cp-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .cp-muted { color: var(--cp-muted); font-size: .84rem; }
    .cp-response {
        min-height: 125px; padding: 1rem; border: 1px solid var(--cp-border);
        border-radius: 12px; background: rgba(8,20,37,.68); color: #e5edf7;
        white-space: pre-wrap; overflow-wrap: anywhere;
    }
    div[data-testid="stForm"] {
        padding: 1.15rem 1.25rem 1.3rem; border: 1px solid var(--cp-border);
        border-radius: 16px; background: rgba(14,32,55,.56);
    }
    div[data-testid="stMetric"] {
        padding: .8rem 1rem; border: 1px solid var(--cp-border);
        border-radius: 12px; background: rgba(14,32,55,.66);
    }
    .stButton > button, [data-testid="stFormSubmitButton"] button {
        min-height: 2.75rem; border-radius: 10px; font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


ACTION_PRESENTATION = {
    EnforcementAction.ALLOW: ("Allowed", "#34d399", "The response passed the active policy."),
    EnforcementAction.WARN: ("Allowed with warning", "#fbbf24", "The response may be shown with a safety notice."),
    EnforcementAction.REDACT: ("Sensitive data redacted", "#38bdf8", "Detected sensitive spans were removed."),
    EnforcementAction.REVIEW: ("Human review required", "#f59e0b", "The response is being held for accountable review."),
    EnforcementAction.BLOCK: ("Blocked", "#fb7185", "The response must not be released."),
}
SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


@st.cache_resource
def build_checker() -> ControlPlane:
    return ControlPlane(verification_service=build_adaptive_verification_from_env())


def category_summary(report) -> list[dict]:
    rows = []
    for category in ("privacy", "bias", "hallucination", "policy"):
        matches = [finding for finding in report.findings if finding.category.value == category]
        highest = max(
            (finding.severity.value for finding in matches),
            key=lambda value: SEVERITY_ORDER[value],
            default="clear",
        )
        rows.append(
            {
                "Category": category.replace("_", " ").title(),
                "Findings": len(matches),
                "Highest severity": highest.title(),
                "Max confidence": max((finding.confidence for finding in matches), default=None),
                "Status": ", ".join(sorted({finding.status.value for finding in matches})) or "clear",
            }
        )
    return rows


def adaptive_result_from(report):
    return next((result for result in report.detector_results if result.detector == "adaptivefact"), None)


checker = build_checker()
verification_service = checker.verification_service
profiles = checker.policy_repository.available()

st.markdown(
    """
    <section class="cp-hero">
        <div class="cp-eyebrow">Responsible AI gateway</div>
        <h1>ControlPlane.ai</h1>
        <p>Inspect an AI response for privacy, bias, factuality, and conversation risk before it reaches a user.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Control center")
    profile = st.selectbox(
        "Policy profile",
        profiles,
        format_func=lambda value: value.replace("_", " ").title(),
        help="The policy determines enabled checks, verification depth, and enforcement actions.",
    )
    selected_policy = checker.policy_repository.load(profile)
    consequential = st.toggle(
        "Consequential use case",
        help="Enable for decisions with meaningful impact on a person or organization.",
    )
    st.markdown("---")
    if verification_service is None:
        st.markdown(
            '<span class="cp-status"><span class="cp-dot" style="background:#fbbf24"></span>Deterministic verification</span>',
            unsafe_allow_html=True,
        )
        st.caption("AdaptiveFact/NLI was explicitly disabled in the environment.")
    else:
        st.markdown(
            '<span class="cp-status"><span class="cp-dot" style="background:#34d399"></span>AdaptiveFact enabled</span>',
            unsafe_allow_html=True,
        )
        st.caption(os.getenv("CONTROLPLANE_NLI_MODEL", "cross-encoder/nli-deberta-v3-small"))

    hallucination_policy = selected_policy.checks.get("hallucination")
    requested_depth = hallucination_policy.depth.value if hallucination_policy else "quick"
    st.markdown("#### Active policy")
    st.caption(selected_policy.description)
    st.markdown(
        f"""
        <div class="cp-muted">
            Version <b>{html.escape(selected_policy.version)}</b><br>
            Verification <b>{html.escape(requested_depth.title())}</b><br>
            Latency budget <b>{selected_policy.latency_budget_ms:,} ms</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("ControlPlane produces safety signals, not legal or compliance certification.")

st.markdown("### Inspect an interaction")
st.caption("Provide the generated response and, when available, the evidence it should be grounded in.")

with st.form("interaction_form", clear_on_submit=False):
    prompt = st.text_area(
        "User prompt",
        "Can I share the customer's account details?",
        height=90,
        placeholder="What did the user ask?",
    )
    left_input, right_input = st.columns(2, gap="large")
    with left_input:
        response = st.text_area(
            "AI response",
            "Contact the customer at alex@example.com. Their account balance is $12,000.",
            height=190,
            placeholder="Paste the model response to inspect…",
        )
    with right_input:
        context = st.text_area(
            "Grounding evidence · optional",
            "The customer requested a callback. Do not reveal private identifiers.",
            height=190,
            placeholder="Paste trusted documents, retrieved passages, or source context…",
            help="Factuality checks compare claims against this evidence.",
        )
    submitted = st.form_submit_button("Run safety check", type="primary", use_container_width=True)

if submitted:
    if not prompt.strip():
        st.error("Enter a user prompt before running the check.")
    elif not response.strip():
        st.error("Enter an AI response before running the check.")
    else:
        with st.spinner("Running policy checks and factuality verification…"):
            st.session_state["latest_report"] = checker.check(
                Interaction(
                    profile=profile,
                    prompt=prompt,
                    response=response,
                    context=context.strip() or None,
                    consequential=consequential,
                )
            )

report = st.session_state.get("latest_report")
if report is None:
    st.info("Configure the interaction above, then run a safety check to see the policy decision and evidence trail.")
    st.stop()

adaptive_result = adaptive_result_from(report)
action_title, action_color, action_note = ACTION_PRESENTATION[report.decision.action]
st.markdown("---")
st.markdown("## Decision")
st.markdown(
    f"""
    <div class="cp-decision" style="--decision-color:{action_color}">
        <div class="cp-decision-title">{html.escape(action_title)}</div>
        <div class="cp-decision-reason">{html.escape(report.decision.reason)} {html.escape(action_note)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if adaptive_result is None:
    factuality_status = "Deterministic only"
elif adaptive_result.error:
    factuality_status = "Verification failed"
else:
    factuality_status = str(adaptive_result.metadata.get("response_status", "unknown")).replace("_", " ").title()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Policy action", report.decision.action.value.replace("_", " ").title())
metric_2.metric("Policy risk index", f"{report.decision.risk_score:.2f}")
metric_3.metric("Factuality", factuality_status)
metric_4.metric("Total latency", f"{report.total_latency_ms:,.1f} ms")
st.caption("The policy risk index reflects the strongest finding; it is not a probability that the response is false.")

overview_tab, findings_tab, claims_tab, audit_tab = st.tabs(
    ["Overview", f"Findings ({len(report.findings)})", "Claim verification", "Audit details"]
)

with overview_tab:
    original_col, controlled_col = st.columns(2, gap="large")
    with original_col:
        st.markdown("#### Original response")
        st.markdown(f'<div class="cp-response">{html.escape(report.original_response)}</div>', unsafe_allow_html=True)
    with controlled_col:
        st.markdown("#### Policy-controlled response")
        st.markdown(f'<div class="cp-response">{html.escape(report.final_response)}</div>', unsafe_allow_html=True)

    st.markdown("#### Risk coverage")
    st.dataframe(
        category_summary(report),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Max confidence": st.column_config.ProgressColumn(
                "Max confidence", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )
    if report.decision.matched_rules:
        st.markdown("**Matched policy rules:** " + ", ".join(f"`{rule}`" for rule in report.decision.matched_rules))
    else:
        st.caption("No policy rule matched; the profile's default action was used.")

with findings_tab:
    if not report.findings:
        st.success("No reportable risks were detected.")
    else:
        st.dataframe(
            [
                {
                    "Category": finding.category.value.title(),
                    "Subtype": finding.subtype.replace("_", " ").title(),
                    "Severity": finding.severity.value.title(),
                    "Confidence": finding.confidence,
                    "Status": finding.status.value.title(),
                    "Message": finding.message,
                    "Detector": finding.detector,
                }
                for finding in report.findings
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                )
            },
        )
        for finding in report.findings:
            if finding.evidence:
                with st.expander(f"Evidence · {finding.subtype.replace('_', ' ').title()}"):
                    for item in finding.evidence:
                        st.code(item, language=None)

with claims_tab:
    if adaptive_result is None:
        st.warning("AdaptiveFact/NLI did not run because it was explicitly disabled. Restart Streamlit after enabling it.")
    elif adaptive_result.error:
        st.error(f"Adaptive verification failed: {adaptive_result.error}")
    else:
        phase6 = adaptive_result.metadata.get("phase6", {})
        st.caption(
            f"Depth: {adaptive_result.metadata.get('verification_depth', 'unknown')} · "
            f"Claims: {adaptive_result.metadata.get('claims_checked', 0)} · "
            f"NLI claims: {phase6.get('nli_claims', 0)} · Evidence pairs: {phase6.get('nli_pairs', 0)}"
        )
        claim_rows = []
        for claim in adaptive_result.metadata.get("claim_results", []):
            scores = claim.get("nli_scores", {})
            claim_rows.append(
                {
                    "Claim": claim.get("claim"),
                    "Status": str(claim.get("status", "unknown")).title(),
                    "Confidence": claim.get("confidence"),
                    "Verifier": claim.get("verifier"),
                    "Entailment": scores.get("entailment"),
                    "Contradiction": scores.get("contradiction"),
                    "Evidence": " | ".join(claim.get("evidence", [])),
                }
            )
        if claim_rows:
            st.dataframe(claim_rows, hide_index=True, use_container_width=True)
        else:
            st.info("No verifiable factual claims were extracted from this response.")

with audit_tab:
    audit_col, detector_col = st.columns(2)
    with audit_col:
        st.markdown("#### Request trace")
        st.write(f"**Check ID:** `{report.id}`")
        st.write(f"**Interaction ID:** `{report.interaction_id}`")
        st.write(f"**Policy:** `{report.policy_id}` · version `{report.policy_version}`")
        st.write(f"**Created:** {report.created_at.isoformat()}")
        st.write(f"**Audit event:** `{report.audit_event_id or 'audit disabled'}`")
    with detector_col:
        st.markdown("#### Detector execution")
        for result in report.detector_results:
            if result.error:
                st.error(f"{result.detector}: {result.error}")
            elif result.skipped:
                st.warning(f"{result.detector}: skipped — {result.skip_reason or 'no reason provided'}")
            else:
                st.write(f"**{result.detector.title()}** · {result.latency_ms:,.1f} ms · {len(result.findings)} finding(s)")
    with st.expander("Raw JSON report"):
        st.json(report.model_dump(mode="json"))
