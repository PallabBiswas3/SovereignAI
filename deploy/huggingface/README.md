---
title: SovereignAI GPU Model
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: '3.10'
pinned: false
models:
  - Qwen/Qwen3-4B-Instruct-2507
---

Private ZeroGPU text inference for the owner's local SovereignAI application.

The model is Qwen3 4B Instruct. The Gradio API exposes `/generate` for streaming
text/structured JSON and `/runtime_info` for non-generating connection checks.
Application documents, tools and databases are not uploaded with this Space.

This Space needs **ZeroGPU** hardware (`zero-a10g`); CPU hosting is not a GPU
substitute. Free GPU availability depends on account eligibility and daily quota.
The deployment helper never selects paid hardware or upgrades an account.

Protocol marker: `sovereign-hf-v1`.
