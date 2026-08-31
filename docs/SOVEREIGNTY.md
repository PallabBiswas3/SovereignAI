# Sovereignty verification

SovereignAI distinguishes two different claims that must not be conflated.

## Application-level controls

All inference adapters validate destinations before connecting. Only loopback addresses and the explicit Compose service names `backend`, `frontend`, `ollama`, `qdrant`, `sandbox`, and `ocr` are accepted. Public hostnames and private-LAN IP addresses are rejected. The monitor API reports configured endpoints and application-controlled blocked attempts.

This proves what the application is configured and coded to do. It does **not** prove that the host OS, an imported library, an administrator, or an unrelated process cannot access the internet.

## Network-level isolation

The Compose `sovereign` network is declared `internal: true`. Backend, frontend, optional Qdrant, and the optional pinned Ollama container communicate on that internal network. Published ports bind only to `127.0.0.1`. The sandbox separately uses `network_mode: none`.

Prepare images and model weights while connected, then start the isolated stack:

```powershell
docker compose --profile container-ollama pull
docker compose --profile sandbox-build build
docker compose --profile container-ollama up -d ollama backend frontend
docker exec sovereign-ai-backend-1 python scripts/verify_airgap.py --require-ollama
```

The verifier makes two active probes from the backend network namespace. It succeeds only when the public probe fails and, with `--require-ollama`, the internal Ollama `/api/tags` endpoint succeeds. Container naming can vary; use `docker compose ps` to find the backend name.

If the backend runs directly on the host to access the Docker sandbox, enforce equivalent outbound firewall rules and run the verifier from that same account/network namespace. A successful application policy check alone is not a network accreditation.

## Offline staging checklist

1. Pin and acquire container images, Python wheels, npm packages, the embedding model, and both Ollama model blobs on an approved staging system.
2. Record hashes and scan all transferred material.
3. Import artifacts and populate the `ollama-models` volume before disconnection.
4. Disable external interfaces or apply deny-by-default egress rules.
5. Run the active verifier and retain its JSON output with deployment evidence.
6. Run the offline test and evaluation suites.

Known limitation: this repository does not provide packet capture, signed attestation, firewall management, or a formal security accreditation. Those are deployment responsibilities.
