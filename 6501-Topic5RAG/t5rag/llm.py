from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer

from .device import DeviceType


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.3


class LocalLLM:
    def __init__(
        self,
        model_name: str,
        device: DeviceType,
        dtype: "object",  # torch.dtype
        tokenizer: "object",
        model: "object",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def load(cls, model_name: str, device: DeviceType, dtype: "object") -> "LocalLLM":
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        # Put model in eval mode; we are not training.
        model.eval()

        return cls(
            model_name=model_name,
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            model=model,
        )

    def generate(self, prompt: str, cfg: GenerationConfig | None = None) -> str:
        import torch

        cfg = cfg or GenerationConfig()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                do_sample=True if cfg.temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens.
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return response.strip()
