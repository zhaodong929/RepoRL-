"""Strict TOML configuration models for cloud training jobs."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TypeVar

from pydantic import Field

from reporl.schemas import StrictModel


class LoRAConfig(StrictModel):
    rank: int = Field(default=32, ge=1)
    alpha: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


class SFTConfig(StrictModel):
    model_id: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    model_revision: str = "main"
    train_file: Path
    eval_file: Path | None = None
    output_dir: Path
    max_length: int = Field(default=8_192, ge=512)
    epochs: float = Field(default=2.0, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    per_device_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    weight_decay: float = Field(default=0.0, ge=0)
    logging_steps: int = Field(default=5, ge=1)
    save_steps: int = Field(default=100, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    seed: int = 42
    load_in_4bit: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = True
    resume_from_checkpoint: Path | None = None
    lora: LoRAConfig = Field(default_factory=LoRAConfig)


class GRPOConfig(StrictModel):
    model_id: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    model_revision: str = "main"
    initial_adapter: Path
    rollout_groups_file: Path
    output_dir: Path
    expected_policy_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    epochs: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=5e-6, gt=0)
    clip_epsilon: float = Field(default=0.2, gt=0, lt=1)
    kl_beta: float = Field(default=0.02, ge=0)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    initial_log_ratio_tolerance: float = Field(default=0.1, gt=0, le=1)
    seed: int = 42
    load_in_4bit: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = True


ConfigT = TypeVar("ConfigT", bound=StrictModel)


def load_toml_config(path: Path, model: type[ConfigT], *, section: str) -> ConfigT:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if section not in payload:
        raise ValueError(f"configuration file is missing [{section}]")
    return model.model_validate(payload[section])
