"""Lazy Hugging Face policy backend for GPU rollout workers."""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reporl.agent.models import ChatMessage, PolicyIdentity, PolicyStep
from reporl.agent.policy import PolicyContextLengthError
from reporl.schemas import GenerationTrace, TokenUsage
from reporl.tasks.canonical import artifact_sha256


class TransformersPolicy:
    """Generate one action at a time while retaining exact behavior-policy logprobs."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        identity: PolicyIdentity,
        generation_config: Any,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._identity = identity
        self._generation_config = generation_config
        self._model_id = identity.model_id
        self._revision = identity.digest
        self._max_input_tokens = identity.max_input_tokens
        self._max_new_tokens = identity.max_new_tokens
        self._temperature = identity.sampling_temperature
        self._top_p = identity.sampling_top_p
        self._model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str = "main",
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
        torch_dtype: str = "bfloat16",
        attn_implementation: str | None = "sdpa",
        max_input_tokens: int = 16_384,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> TransformersPolicy:
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "TransformersPolicy requires the 'training' optional dependencies"
            ) from error

        dtype = getattr(torch, torch_dtype)
        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "torch_dtype": dtype,
            "device_map": "auto",
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        if load_in_4bit:
            model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            use_fast=True,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        resolved_model_revision = str(getattr(model.config, "_commit_hash", None) or revision)
        tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
        resolved_tokenizer_revision = str(
            tokenizer_kwargs.get("_commit_hash") or resolved_model_revision
        )
        adapter_sha256: str | None = None
        model_preparation = "inference-only"
        if adapter_path is not None:
            try:
                peft = importlib.import_module("peft")
            except ModuleNotFoundError as error:
                raise RuntimeError("loading an adapter requires PEFT") from error
            if load_in_4bit:
                model = peft.prepare_model_for_kbit_training(
                    model,
                    use_gradient_checkpointing=True,
                )
                model_preparation = "peft-kbit-training-v1"
            model = peft.PeftModel.from_pretrained(model, adapter_path)
            adapter_sha256 = "sha256:" + directory_sha256(Path(adapter_path))
        chat_template = str(getattr(tokenizer, "chat_template", None) or "")
        special_token_ids = tuple(
            sorted(
                (
                    ("bos", getattr(tokenizer, "bos_token_id", None)),
                    ("eos", getattr(tokenizer, "eos_token_id", None)),
                    ("pad", getattr(tokenizer, "pad_token_id", None)),
                    ("unk", getattr(tokenizer, "unk_token_id", None)),
                )
            )
        )
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        sampling_config: dict[str, Any] = {
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token_id": pad_token_id,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_k": 0,
            "top_p": top_p if temperature > 0 else 1.0,
            "temperature": temperature if temperature > 0 else 1.0,
            "repetition_penalty": 1.0,
            "renormalize_logits": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        generation_config = transformers.GenerationConfig(**sampling_config)
        generation_config_payload = json.dumps(
            generation_config.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        identity = PolicyIdentity(
            model_id=model_id,
            model_revision=resolved_model_revision,
            tokenizer_revision=resolved_tokenizer_revision,
            tokenizer_class=type(tokenizer).__name__,
            adapter_sha256=adapter_sha256,
            quantization="bnb-nf4-double" if load_in_4bit else "none",
            model_preparation=model_preparation,
            torch_dtype=torch_dtype,
            attention_implementation=attn_implementation,
            transformers_version=str(transformers.__version__),
            chat_template_sha256=(
                "sha256:" + hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
            ),
            generation_config_sha256=(
                "sha256:" + hashlib.sha256(generation_config_payload).hexdigest()
            ),
            special_token_ids=special_token_ids,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            sampling_temperature=temperature,
            sampling_top_p=top_p,
            sampling_top_k=0,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            identity=identity,
            generation_config=generation_config,
        )

    @property
    def policy_id(self) -> str:
        return self._model_id

    @property
    def policy_revision(self) -> str:
        return self._revision

    @property
    def policy_identity(self) -> PolicyIdentity:
        return self._identity

    @property
    def adapter_sha256(self) -> str | None:
        return self._identity.adapter_sha256

    def act(
        self,
        messages: Sequence[ChatMessage],
        *,
        seed: int,
        timeout_seconds: float | None = None,
    ) -> PolicyStep:
        del timeout_seconds  # In-process CUDA generation requires an outer process supervisor.
        torch = importlib.import_module("torch")
        wire_messages = []
        for message in messages:
            if message.role == "tool":
                wire_messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL_OBSERVATION\n{message.content}",
                    }
                )
            else:
                wire_messages.append({"role": message.role, "content": message.content})
        encoded = self._tokenizer.apply_chat_template(
            wire_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if encoded.shape[-1] > self._max_input_tokens:
            raise PolicyContextLengthError(
                f"policy context has {encoded.shape[-1]} tokens, limit is {self._max_input_tokens}"
            )
        device = next(self._model.parameters()).device
        prompt_ids = encoded.to(device)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        started = time.monotonic()
        do_sample = self._temperature > 0
        with torch.inference_mode():
            output = self._model.generate(
                input_ids=prompt_ids,
                generation_config=self._generation_config,
            )
            sequence = output.sequences
            generated = sequence[:, prompt_ids.shape[-1] :]
            if len(output.scores) != generated.shape[-1]:
                raise RuntimeError("generation scores do not align with generated tokens")
            if do_sample:
                generated_logprobs = []
                for token_id, scores in zip(generated[0], output.scores, strict=True):
                    row = scores[0].float()
                    selected = row[token_id]
                    generated_logprobs.append(selected - torch.logsumexp(row, dim=-1))
            else:
                generated_logprobs = [
                    torch.zeros((), dtype=torch.float32, device=device)
                    for _ in range(generated.shape[-1])
                ]
        latency_ms = round((time.monotonic() - started) * 1_000)
        generated_ids = tuple(int(token) for token in generated[0].detach().cpu().tolist())
        prompt_id_values = tuple(int(token) for token in prompt_ids[0].detach().cpu().tolist())
        old_logprobs = tuple(float(value.detach().cpu()) for value in generated_logprobs)
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return PolicyStep(
            raw_output=text,
            token_usage=TokenUsage(
                input_tokens=len(prompt_id_values),
                output_tokens=len(generated_ids),
            ),
            latency_ms=latency_ms,
            generation_trace=GenerationTrace(
                prompt_input_ids=prompt_id_values,
                generated_token_ids=generated_ids,
                old_logprobs=old_logprobs,
                sampling_temperature=self._temperature,
                sampling_top_p=self._top_p if do_sample else 1.0,
            ),
        )


def directory_sha256(path: Path) -> str:
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("adapter_path must be a directory")
    digest, _ = artifact_sha256(root)
    return digest.removeprefix("sha256:")
