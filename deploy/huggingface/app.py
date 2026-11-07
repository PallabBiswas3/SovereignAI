"""Private ZeroGPU text generation for SovereignAI. No documents or tools live here."""
from __future__ import annotations

import json
import os
from queue import Empty
from threading import Event, Thread
import time

import gradio as gr
import jsonschema
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, TextIteratorStreamer


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
PROTOCOL = "sovereign-hf-v1"
os.environ.setdefault("DO_NOT_TRACK", "1")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# ZeroGPU requires module-level CUDA placement, before a GPU function is called.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True,
).to("cuda").eval()


def runtime_info():
    return {"model": MODEL_ID, "protocol": PROTOCOL}


class BoundedGeneration(StoppingCriteria):
    def __init__(self, cancelled, deadline):
        self.cancelled, self.deadline = cancelled, deadline

    def __call__(self, input_ids, scores, **kwargs):
        return self.cancelled.is_set() or time.monotonic() >= self.deadline


@spaces.GPU(duration=60)
def generate(prompt, system, max_tokens, schema_json):
    if not isinstance(prompt, str) or not 0 < len(prompt) <= 50000 or len(system) > 50000:
        raise gr.Error("The prompt is empty or exceeds the request limit")
    max_tokens = int(max_tokens)
    if not 1 <= max_tokens <= 1024:
        raise gr.Error("Use an output limit between 1 and 1024 tokens")
    if len(schema_json) > 50000:
        raise gr.Error("The requested schema exceeds the request limit")
    schema = json.loads(schema_json) if schema_json else None
    if schema is not None:
        jsonschema.Draft202012Validator.check_schema(schema)
        system = (system + "\nReturn only a JSON object matching this schema: " + schema_json).strip()
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt", return_dict=True,
    ).to("cuda")
    input_count = encoded["input_ids"].shape[-1]
    if input_count + max_tokens > 8192:
        raise gr.Error("Prompt plus output exceeds the Space's 8192-token limit")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=2)
    cancelled = Event()
    result = {}
    kwargs = {**encoded, "max_new_tokens": max_tokens, "do_sample": False, "streamer": streamer,
              "pad_token_id": tokenizer.eos_token_id, "max_time": 45,
              "stopping_criteria": [BoundedGeneration(cancelled, time.monotonic() + 45)]}
    if schema is not None:
        kwargs["prefix_allowed_tokens_fn"] = build_transformers_prefix_allowed_tokens_fn(
            tokenizer, JsonSchemaParser(schema))

    def work():
        try:
            with torch.inference_mode():
                result["ids"] = model.generate(**kwargs)
        except Exception:
            result["failed"] = True
            streamer.on_finalized_text("", stream_end=True)

    thread = Thread(target=work, daemon=True)
    thread.start()
    text = ""
    try:
        while True:
            try:
                text += next(streamer)
            except Empty:
                if result.get("failed") or not thread.is_alive():
                    break
                continue
            except StopIteration:
                break
            yield {"text": text, "model": MODEL_ID, "done": False}
        thread.join(timeout=10)
        if result.get("failed") or "ids" not in result or thread.is_alive():
            raise gr.Error("GPU generation failed; retry after checking the Space")
        if not text.strip():
            raise gr.Error("The model returned no answer")
        if schema is not None:
            try:
                data = json.loads(text)
                jsonschema.validate(data, schema)
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object")
            except (ValueError, jsonschema.ValidationError):
                raise gr.Error("The model did not complete a valid structured response") from None
        yield {"text": text, "model": MODEL_ID, "done": True,
               "usage": {"prompt_tokens": input_count,
                         "completion_tokens": result["ids"].shape[-1] - input_count}}
    finally:
        cancelled.set()
        thread.join(timeout=10)


with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown("# SovereignAI GPU model\nPrivate text inference for your local application.")
    prompt = gr.Textbox(label="Question")
    system = gr.Textbox(label="System instructions", visible=False)
    max_tokens = gr.Number(value=512, visible=False)
    schema = gr.Textbox(value="", visible=False)
    output = gr.JSON(label="Answer")
    run = gr.Button("Generate")
    run.click(generate, [prompt, system, max_tokens, schema], output,
              api_name="generate", concurrency_limit=1)
    status = gr.Button("Model information", visible=False)
    status.click(runtime_info, [], gr.JSON(visible=False), api_name="runtime_info", queue=False)

demo.queue(max_size=4, default_concurrency_limit=1)
if __name__ == "__main__":
    demo.launch(show_error=False)
