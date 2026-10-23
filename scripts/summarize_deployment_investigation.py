"""Publish a BLOCKED evidence summary; source tests alone cannot release a build."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "local_testing/deployment-remediation-20261023"


def main():
    commands = [json.loads(line) for line in (EVIDENCE / "commands.jsonl").read_text().splitlines()]
    baseline = ROOT / "local_testing/readiness-baseline-20261023T150845Z"
    def counts(path):
        suites = ET.parse(path).getroot().findall(".//testsuite")
        values = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
        values["passed"] = values["tests"] - values["failures"] - values["errors"] - values["skipped"]
        return values
    previous = json.loads((ROOT / "local_testing/production-readiness-20261023/final-source.json").read_text())["source_file_sha256"]
    protected = ["config/access.yaml", "config/policies.yaml", "config/tools.yaml", "deploy/tools.production.yaml",
                 "backend/app/identity/authorization.py", "backend/app/identity/dependencies.py",
                 "backend/app/integrations/orchestrator.py", "Controlplane.ai/src/controlplane/checker.py",
                 "Controlplane.ai/src/controlplane/verification.py", "Time-Series-Diagnostic-Agent/src/tsdiag/domains/process_sensitivity.py"]
    unchanged = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == previous[path] for path in protected}
    published = []
    for command in commands:
        entry = dict(command)
        entry["command"] = [str(arg).replace(str(ROOT), "${REPO_ROOT}").replace(ROOT.as_posix(), "${REPO_ROOT}") for arg in entry["command"]]
        entry["cwd"] = str(command["cwd"]).replace(str(ROOT), "${REPO_ROOT}").replace(ROOT.as_posix(), "${REPO_ROOT}")
        entry["evidence_hash_valid"] = hashlib.sha256((ROOT / command["output"]).read_bytes()).hexdigest() == command["sha256"]
        published.append(entry)
    report = {
        "date": "2026-10-23", "release_verdict": "BLOCKED",
        "baseline_commit": "5eaf907f9b74e686ec49d1aa61755fd0a5f86b0a", "worktree": "uncommitted development changes",
        "confirmed_environmental_findings": [
            "Windows host completed nine certificate-verifying HTTPS probes; Docker large transfers failed in pristine images",
            "Independent Ubuntu WSL reproduced a Debian bad-record-MAC failure, localizing the failure below SovereignAI and beyond Docker alone",
            "Alternate authenticated ciphers, software OpenSSL, host networking and MTU 1280 did not remediate Docker TLS failures",
            "Reversible virtual-adapter RSC test did not remediate failures; original flags restored",
            "Ollama image pull reported containerd ingest I/O failure; Docker VM log recorded virtual-block-device and ext4 journal I/O errors",
            "Docker restart timed out, stop/start recovery did not restore a usable engine; current build attempt received daemon HTTP 500",
            "Host free space about 2.1 GiB; model provisioning refused because it required at least 3845338077 bytes including reserve"
        ],
        "causal_limits": [
            "Exact Windows/WSL driver or component causing TLS stream corruption has not been conclusively identified",
            "Installed network filters are observations, not established causes; none was disabled",
            "Later virtual-disk errors and low space do not prove the cause of the earlier TLS corruption",
            "Containerd write error occurred during the Docker recovery/restart window; no destructive filesystem repair was attempted"
        ],
        "repository_defects_remediated": [
            "Model provisioning resolved mutable revisions and accepted existing directories without content verification",
            "Preflight checked only the presence of model files, not pinned revision or file hashes",
            "Readiness could raise on malformed model-list records rather than report unready",
            "Provisioning lacked a free-space guard",
            "APT update did not explicitly fail on every repository-update error"
        ],
        "models": {"lock": "deploy/model-lock.json", "all_files_verified": True, "offline_model_load_smoke": "passed",
                   "reranker_revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a"},
        "ollama": {"lock": "deploy/ollama-model-lock.json", "official_and_cached_manifests_match": True,
                   "host_blob_hashes_verified": 9, "host_models_verified": 2,
                   "persistent_volume_recreation": "BLOCKED", "runtime_inference": "NOT_RUN",
                   "provisioning_attempt_exit_code": 2, "existing_host_cache_modified": False},
        "tests": {"maintained_python": counts(baseline / "python.xml"), "controlplane_local": counts(baseline / "controlplane.xml"),
                  "frontend": {"passed": 41}, "frontend_typecheck": "passed", "graphrag_build": "passed", "graphrag_auth": {"passed": 3},
                  "supplemental_provisioning_regression": {"passed": 12, "evidence": "local_testing/deployment-remediation-20261023/final-provisioning-regression.log",
                      "scope": "After final helper change; five new fail-closed cases plus seven lock integrity cases. Not a deployed-volume test."},
                  "legacy_failure": "test_pre_authority_ablation_reuses_same_evidence_and_semantic_score; expectation and runtime policy unchanged"},
        "authenticated_deployment_acceptance": [{"scenario": name, "status": "NOT_RUN", "reason": "Complete images/runtime unavailable"} for name in (
            "trusted_tls_and_browser_login", "role_and_workspace_boundaries", "browser_upload_and_task_continuation",
            "diagnosis_authorized_retrieval_and_governed_release", "approval_accept_reject_and_execution_receipts",
            "report_download_and_capsule_verification", "required_service_outage_and_readiness",
            "restart_recovery_and_replay_prevention", "ollama_volume_recreation_and_digest_match")],
        "security_source_unchanged_from_previous_batch": unchanged,
        "all_protected_sources_unchanged": all(unchanged.values()),
        "commands": published,
        "remaining_blockers": ["WSL large-transfer corruption; exact upstream component unresolved", "Docker engine/VHD health and host storage headroom",
                               "Complete image build", "Persisted Ollama volume and real daemon verification", "Authenticated deployment acceptance",
                               "One reproduced legacy ControlPlane ablation test failure"],
        "cleanup": {"rsc_original_flags_restored": True, "application_stack_started": False,
                    "docker_factory_reset_or_data_prune": False, "disposable_mtu_network_removal": "attempted; engine unavailable"}
    }
    (ROOT / "deploy/remediation-20261023.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"release_verdict": report["release_verdict"], "commands": len(published), "protected_sources_unchanged": all(unchanged.values()),
                      "evidence_hashes_valid": all(item["evidence_hash_valid"] for item in published)}))
    return int(not all(unchanged.values()) or not all(item["evidence_hash_valid"] for item in published))


if __name__ == "__main__":
    raise SystemExit(main())
