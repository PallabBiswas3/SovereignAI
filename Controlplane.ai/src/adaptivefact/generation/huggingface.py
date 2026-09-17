from __future__ import annotations

from time import perf_counter

from adaptivefact.generation.base import ChatGenerator, GenerationResult


class HuggingFaceChatGenerator(ChatGenerator):
    """Local causal-LM backend loaded lazily so the rest of the repo stays lightweight.

    A small instruct model can be used for smoke runs; the model id is fully configurable.
    On a CUDA machine, device_map="auto" lets Transformers place the model automatically.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device_map: str | None = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceChatGenerator requires transformers and torch. "
                "Install the Phase 1 dependencies from requirements.txt."
            ) from exc

        self.model_name = model_name
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        kwargs = {"trust_remote_code": trust_remote_code}
        if device_map is not None:
            kwargs["device_map"] = device_map
        if torch_dtype == "auto":
            kwargs["torch_dtype"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> GenerationResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        start = perf_counter()
        if hasattr(self.tokenizer, "apply_chat_template"):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt")

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = int(inputs["input_ids"].shape[-1])

        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)

        generated_ids = output[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        latency_ms = (perf_counter() - start) * 1000.0

        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            input_tokens=input_len,
            output_tokens=int(generated_ids.shape[-1]),
        )
