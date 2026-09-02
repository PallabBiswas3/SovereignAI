import asyncio
from pathlib import Path

from app.monitoring import network
from app.monitoring.network import AirGapVerifier, LocalNetworkPolicy


ROOT = Path(__file__).resolve().parents[1]


def test_configured_inference_is_air_gap_safe() -> None:
    result = AirGapVerifier().verify_model_config(ROOT / "config" / "models.yaml")
    assert result["passed"]
    assert not result["violations"]


def test_external_destination_is_blocked() -> None:
    assert LocalNetworkPolicy.is_local_url("http://127.0.0.1:11434")
    assert LocalNetworkPolicy.is_local_url("http://qdrant:6333")
    assert not LocalNetworkPolicy.is_local_url("https://api.openai.com/v1")
    assert not LocalNetworkPolicy.is_local_url("http://192.168.1.20:8080")
    try:
        LocalNetworkPolicy.require_local("https://api.openai.com/v1")
    except ValueError as exc:
        assert "blocked" in str(exc).lower()
    else:
        raise AssertionError("External destination was permitted")


def test_monitor_does_not_crash_when_docker_probe_is_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(
        network.shutil,
        "which",
        lambda command: "docker.exe" if command == "docker" else None,
    )

    def unsupported_subprocess(*args, **kwargs):
        raise NotImplementedError

    monkeypatch.setattr(network.subprocess, "run", unsupported_subprocess)
    services = asyncio.run(network.local_service_status("http://127.0.0.1:9"))
    sandbox = next(service for service in services if service["name"] == "Sandbox")
    assert sandbox["status"] == "unavailable"


def test_compose_declares_network_isolation_and_internal_ollama() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    verifier = (ROOT / "backend" / "scripts" / "verify_airgap.py").read_text(encoding="utf-8")
    assert "internal: true" in compose
    assert "ollama/ollama:0.33.2" in compose
    assert "external_egress_blocked" in verifier
    assert "ollama_reachable" in verifier
