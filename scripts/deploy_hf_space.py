"""Create a private FREE ZeroGPU Space, verify a reply, and connect the app.

Enter your Hugging Face write token in the hidden local terminal prompt.
Only deploy/huggingface/{app.py,requirements.txt,README.md} are uploaded.
No benchmark, paid hardware, subscription changes or project uploads.
"""
from __future__ import annotations

import argparse
import asyncio
from getpass import getpass
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from huggingface_hub import HfApi, SpaceHardware, get_token
from app.llm.hf_space_provider import HuggingFaceSpaceProvider, validate_space_url

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


class SetupError(RuntimeError):
    """Fixed, actionable messages authored by this script, safe for the terminal."""


def safe_hub_message(error, token):
    """Extract only the server's error field, without headers, URLs or secrets."""
    message = None
    response = getattr(error, "response", None)
    if response is not None:
        try:
            body = response.json()
            if isinstance(body, dict) and isinstance(body.get("error"), str):
                message = body["error"]
        except (ValueError, TypeError):
            pass
    if not message:
        candidate = getattr(error, "server_message", None)
        message = candidate if isinstance(candidate, str) else ""
    if not message:
        return ""
    # Redact before printing or truncating; never display str(error), which can
    # contain request/response details. Also remove terminal control characters.
    message = message.replace(token, "[redacted]") if token else message
    message = re.sub(r"hf_[A-Za-z0-9_-]+", "[redacted]", message)
    message = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [redacted]", message)
    message = re.sub(r"https?://\S+", "[URL omitted]", message)
    message = re.sub(r"\b[^\s@]+@[^\s@]+\.[^\s@]+", "[email omitted]", message)
    message = " ".join("".join(char if char.isprintable() else " " for char in message).split())
    return message[:1000]


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def replace_settings(original, updates):
    remaining, lines = dict(updates), []
    for line in original.splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=", line)
        key = match.group(1) if match else None
        if key in updates:
            if key in remaining:
                lines.append(key + "=" + json.dumps(str(remaining.pop(key))))
        else:
            lines.append(line)
    lines += [key + "=" + json.dumps(str(value)) for key, value in remaining.items()]
    return "\n".join(lines) + "\n"


def save_connection(endpoint, token, *, docker=False, root=ROOT):
    validate_space_url(endpoint)
    target = root / "deploy" / ".env" if docker else root / ".env"
    secret_path = root / "deploy" / "secrets" / "hf_token"
    updates = {"SOVEREIGN_LLM_PROVIDER": "hf_space", "SOVEREIGN_REMOTE_INFERENCE_ENABLED": "true",
               "SOVEREIGN_HF_SPACE_URL": endpoint.rstrip("/"), "SOVEREIGN_HF_SPACE_MODEL": MODEL,
               "SOVEREIGN_HF_TOKEN": "", "SOVEREIGN_HF_TOKEN_FILE": (
                   "/run/secrets/hf_token" if docker else secret_path.as_posix()),
               "SOVEREIGN_VLLM_MAX_TOKENS": "512", "SOVEREIGN_MODEL_GENERATION_TIMEOUT_SECONDS": "180"}
    original = target.read_text(encoding="utf-8") if target.exists() else ""
    atomic_write(secret_path, token + "\n")
    atomic_write(target, replace_settings(original, updates))
    return target


def wait_for_space(api, repo_id, timeout=1200):
    deadline, previous = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        runtime = api.get_space_runtime(repo_id)
        stage = getattr(runtime.stage, "value", runtime.stage)
        if stage != previous:
            print("Space:", stage, flush=True)
            previous = stage
        if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
            raise RuntimeError("Space failed to start")
        if stage == "RUNNING":
            hardware = getattr(runtime.hardware, "value", runtime.hardware)
            if hardware != SpaceHardware.ZERO_A10G.value:
                raise RuntimeError("Space is not running on free ZeroGPU")
            endpoint = getattr(api.space_info(repo_id), "host", None)
            if not endpoint:
                raise RuntimeError("Space endpoint is missing")
            validate_space_url(endpoint)
            return endpoint.rstrip("/")
        time.sleep(10)
    raise TimeoutError("Space is still starting")


async def verify_model(endpoint, token):
    provider = HuggingFaceSpaceProvider(endpoint, enabled=True, api_key=token,
                                        timeout_seconds=180, max_tokens=16)
    health = await provider.health_check()
    if not health.get("available") or MODEL not in {item["id"] for item in health.get("models", [])}:
        raise RuntimeError("Space model API is not ready or the model does not match")
    result = await provider.generate("Reply with OK.", MODEL)
    if not result.text.strip() or result.fallback:
        raise RuntimeError("The GPU model did not answer")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", help="Require this account before creating a Space")
    parser.add_argument("--space-name", default="sovereign-qwen3-gpu")
    parser.add_argument("--existing-space", help="Connect owner/space without uploading or replacing files")
    parser.add_argument("--upload-existing", action="store_true",
                        help="With --existing-space, upload the three inference files to finish an interrupted upload")
    parser.add_argument("--docker", action="store_true", help="Save Docker settings instead of native settings")
    args = parser.parse_args(argv)
    if args.upload_existing and not args.existing_space:
        parser.error("--upload-existing requires --existing-space owner/space")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,80}", args.space_name):
        parser.error("Use a Space name containing letters, digits, hyphens or underscores")
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        if not sys.stdin.isatty():
            parser.error("Run in an interactive terminal for hidden token entry, or set HF_TOKEN privately")
        print("Paste the actual access token beginning with hf_, then press Enter. Nothing will appear while you paste/type.", flush=True)
        token = getpass("Hugging Face write token (hidden; never send it in chat): ").strip()
    token = token.strip()
    if not token:
        parser.error("No token received. Copy the access token from https://huggingface.co/settings/tokens and paste it at the hidden prompt")
    if not token.startswith("hf_") or len(token) <= 3 or any(char.isspace() for char in token):
        parser.error("Expected the full access token beginning with hf_, without quotes or spaces. Enter the token value, not its name, your username, or your password")
    repo_id = None
    operation = "account authentication"
    space_confirmed = False
    needs_upload = False
    try:
        api = HfApi(token=token)
        identity = api.whoami()
        owner = identity["name"]
        if args.username and owner.lower() != args.username.lower():
            raise SetupError("The token belongs to a different account than --username")
        role = ((identity.get("auth") or {}).get("accessToken") or {}).get("role", "unknown")
        role_label = {"read": "read-only", "write": "write", "fineGrained": "fine-grained"}.get(role, "unknown")
        print(f"Authenticated as {owner}. Token permission: {role_label}.", flush=True)
        if role == "read" and (not args.existing_space or args.upload_existing):
            raise SetupError("This token is read-only. Create a write token (or a fine-grained token that allows creating/writing your Space), then rerun setup")
        repo_id = args.existing_space or f"{owner}/{args.space_name}"
        if not re.fullmatch(re.escape(owner) + r"/[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", repo_id):
            raise SetupError("The Space must belong to your authenticated personal account")
        if not args.existing_space:
            operation = "private ZeroGPU Space creation"
            print("Creating a private ZeroGPU Space (no paid hardware)...", flush=True)
            api.create_repo(repo_id=repo_id, repo_type="space", private=True, space_sdk="gradio",
                            space_hardware=SpaceHardware.ZERO_A10G, exist_ok=False)
            space_confirmed = True
            needs_upload = True
        else:
            operation = "existing Space lookup"
            info = api.space_info(repo_id)
            space_confirmed = True
            if getattr(info, "private", None) is not True:
                raise SetupError("The selected Space is not private; choose the intended private Space")
            needs_upload = args.upload_existing
        if needs_upload:
            operation = "inference file upload"
            api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=ROOT / "deploy" / "huggingface",
                              allow_patterns=["app.py", "README.md", "requirements.txt"],
                              commit_message="Add private SovereignAI ZeroGPU model")
            needs_upload = False
        print(f"Space: https://huggingface.co/spaces/{repo_id}", flush=True)
        operation = "Space build and startup"
        print("Waiting for build and model loading; first startup can take several minutes.", flush=True)
        endpoint = wait_for_space(api, repo_id)
        operation = "model connection check"
        print("Checking the model with one short question...", flush=True)
        asyncio.run(verify_model(endpoint, token))
        operation = "local connection settings save"
        target = save_connection(endpoint, token, docker=args.docker)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        suffix = f", HTTP {status}" if isinstance(status, int) else ""
        print(f"Setup stopped during {operation} ({type(exc).__name__}{suffix}). Local settings were not activated.")
        message = safe_hub_message(exc, token)
        if message:
            print("Hugging Face error (redacted): " + message)
        if isinstance(exc, SetupError):
            print(str(exc))  # Only fixed script-authored messages.
        elif status == 401:
            print("Authentication/access was rejected. Check the token's validity and access to this Space.")
        elif status == 402 and operation == "private ZeroGPU Space creation":
            print("Hugging Face rejected free GPU hosting for this account. Changing or recreating tokens will not resolve this eligibility/billing response.")
            print("For a free route, check account email/age/Space allowance and request a community GPU grant if needed: https://huggingface.co/docs/hub/spaces-gpus#community-gpu-grants")
            print("A grant requires Hugging Face approval; it is not an instant GPU allocation. No subscription, paid hardware or public Space was selected.")
        elif status == 403:
            print("Access was forbidden. The server message above may identify token scope, email verification, account eligibility or the ZeroGPU Space limit.")
            print("Free hosting currently requires an eligible personal account with verified email, age over 30 days and a free ZeroGPU slot. No paid alternative was selected.")
        elif status == 409:
            print("Space already exists. Use --existing-space owner/space or another --space-name.")
        else:
            print("Check Space build/runtime logs, GPU quota, network access and account selection.")
        if repo_id and (space_confirmed or status == 409) and not isinstance(exc, SetupError):
            command = ".venv/Scripts/python.exe scripts/deploy_hf_space.py --existing-space " + repo_id
            if needs_upload:
                command += " --upload-existing"
            if args.username:
                command += " --username " + args.username
            if args.docker:
                command += " --docker"
            print("To resume: " + command)
        elif not space_confirmed and status == 402 and operation == "private ZeroGPU Space creation":
            print("No Space was created. Retry only after free-hosting eligibility or a grant is confirmed; --existing-space cannot help yet.")
        elif not space_confirmed:
            print("Space creation was not confirmed. Correct the access issue and rerun the original command; do not use --existing-space unless that Space actually exists.")
        return 1
    print("The GPU model answered. Remote connection saved to " + str(target))
    if args.docker:
        print("Run: docker compose --env-file deploy/.env -f compose.yaml -f compose.hf-space.yaml up -d --build backend frontend")
    else:
        print("Restart your backend to use the Hugging Face GPU model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
