"""Deployment boundary and configurable model contracts; no daemon needed."""
from pathlib import Path

import yaml

from app.router.model_registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[2]


def test_ollama_overrides_keep_optional_roles_and_other_providers(monkeypatch, tmp_path):
    monkeypatch.setenv("SOVEREIGN_OLLAMA_URL", "http://ollama:11434")
    monkeypatch.setenv("SOVEREIGN_OLLAMA_GENERAL_MODEL", "custom-local:4b")
    source = yaml.safe_load((ROOT / "config/models.docker.yaml").read_text())
    source["models"]["coder"]["provider"] = "vllm"
    source["models"]["coder"]["endpoint"] = "http://vllm:8001/v1"
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump(source))
    registry = ModelRegistry(config)
    assert registry.get("general").model_tag == "custom-local:4b"
    assert registry.get("general").endpoint == "http://ollama:11434"
    assert registry.get("coder").endpoint == "http://vllm:8001/v1"
    assert registry.get("vision").model_tag == source["models"]["vision"]["model_tag"]


def test_ollama_has_no_governance_ingress_or_storage_network():
    profile = yaml.safe_load((ROOT / "deploy/compose.production.yml").read_text())
    services = profile["services"]
    ollama_networks = set(services["ollama"]["networks"])
    assert ollama_networks == {"inference"}
    for service in ("controlplane", "diagnostics", "frontend", "gateway", "graph-db", "graph-rest"):
        assert not ollama_networks.intersection(services[service]["networks"])
    assert profile["networks"]["edge"]["internal"] is False
    assert all(network["internal"] for name, network in profile["networks"].items() if name != "edge")
    assert [name for name, config in services.items() if "edge" in config["networks"]] == ["gateway"]
    assert [name for name, config in services.items() if config.get("ports")] == ["gateway"]
    assert all(config["mem_limit"] and config["pids_limit"] for config in services.values())
    assert all(config["security_opt"] == ["no-new-privileges:true"] for config in services.values())


def test_durable_state_and_startup_never_launch_a_task():
    profile = yaml.safe_load((ROOT / "deploy/compose.production.yml").read_text())
    backend = profile["services"]["backend"]
    assert "backend-data:/app/backend/data" in backend["volumes"]
    assert "workspace:/app/workspace" in backend["volumes"]
    assert "controlplane-audit:/app/Controlplane.ai/results" in profile["services"]["controlplane"]["volumes"]
    assert "docker.sock" not in str(profile)
    assert backend["environment"]["SOVEREIGN_INTEGRATION_FAIL_CLOSED"] == "true"
    assert backend["environment"]["SOVEREIGN_ALLOW_DETERMINISTIC_FALLBACK"] == "false"
