"""Single-GPU LoRA GRPO updates over externally collected interactive rollout groups."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import random
from pathlib import Path
from typing import Any

from reporl.agent.hf_policy import directory_sha256
from reporl.tasks.canonical import canonical_sha256
from reporl.training.config import GRPOConfig, load_toml_config
from reporl.training.math import grouped_standardized_advantages
from reporl.training.provenance import (
    artifact_evidence,
    git_state,
    prepare_output_directory,
)
from reporl.training.records import GRPOGroup, read_grpo_groups_jsonl


def _runtime() -> tuple[Any, Any, Any]:
    try:
        peft: Any = importlib.import_module("peft")
        torch: Any = importlib.import_module("torch")
        transformers: Any = importlib.import_module("transformers")
    except ModuleNotFoundError as error:
        raise RuntimeError("GRPO requires RepoRL's 'training' optional dependencies") from error
    return peft, torch, transformers


def _generated_logits(torch: Any, model: Any, trace: Any, device: Any) -> tuple[Any, Any]:
    prompt = list(trace.prompt_input_ids)
    generated = list(trace.generated_token_ids)
    values = torch.tensor([prompt + generated], dtype=torch.long, device=device)
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    parameters = inspect.signature(base_model.forward).parameters
    if "logits_to_keep" in parameters:
        keep_argument = "logits_to_keep"
    elif "num_logits_to_keep" in parameters:
        keep_argument = "num_logits_to_keep"
    else:
        raise RuntimeError(
            "the model does not support selective logits; refusing an OOM-prone full-sequence pass"
        )
    output = model(
        input_ids=values,
        use_cache=False,
        **{keep_argument: len(generated) + 1},
    )
    logits = output.logits
    if logits.shape[1] != len(generated) + 1:
        raise RuntimeError("selective-logits output does not match the requested token window")
    targets = values[0, len(prompt) :]
    return logits[0, : len(generated), :], targets


def _sampled_logprobs(torch: Any, logits: Any, targets: Any, trace: Any) -> Any:
    if trace.sampling_temperature <= 0:
        raise ValueError("GRPO cannot train from deterministic generation traces")
    logprobs = []
    for row, target in zip(logits, targets, strict=True):
        scores = row.float() / trace.sampling_temperature
        if trace.sampling_top_p < 1:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
            sorted_remove = cumulative > trace.sampling_top_p
            sorted_remove[1:] = sorted_remove[:-1].clone()
            sorted_remove[0] = False
            remove = torch.zeros_like(sorted_remove).scatter(
                0,
                sorted_indices,
                sorted_remove,
            )
            scores = scores.masked_fill(remove, float("-inf"))
        selected = scores[target]
        logprobs.append(selected - torch.logsumexp(scores, dim=-1))
    return torch.stack(logprobs).clamp(min=-60.0)


def _generated_logprobs(torch: Any, model: Any, trace: Any, device: Any) -> Any:
    logits, targets = _generated_logits(torch, model, trace, device)
    return _sampled_logprobs(torch, logits, targets, trace)


def _trace_loss(
    *,
    torch: Any,
    model: Any,
    trace: Any,
    advantage: float,
    clip_epsilon: float,
    kl_beta: float,
    device: Any,
) -> tuple[Any, float, float, float]:
    model.set_adapter("reference")
    with torch.no_grad():
        reference_logprobs = _generated_logprobs(torch, model, trace, device)
    model.set_adapter("policy")
    current_logprobs = _generated_logprobs(torch, model, trace, device)
    old_logprobs = torch.tensor(
        trace.old_logprobs,
        dtype=current_logprobs.dtype,
        device=device,
    )
    log_ratio = (current_logprobs - old_logprobs).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    advantage_tensor = torch.full_like(ratio, advantage)
    surrogate = torch.minimum(ratio * advantage_tensor, clipped_ratio * advantage_tensor)
    reference_delta = (reference_logprobs - current_logprobs).clamp(min=-20.0, max=20.0)
    kl = torch.exp(reference_delta) - reference_delta - 1.0
    token_losses = -surrogate + kl_beta * kl
    return (
        token_losses,
        float(kl.detach().sum().cpu()),
        float(ratio.detach().sum().cpu()),
        float(log_ratio.detach().abs().max().cpu()),
    )


def _validate_groups(groups: tuple[GRPOGroup, ...], config: GRPOConfig) -> None:
    if not groups:
        raise ValueError("the rollout group file is empty")
    seen: set[str] = set()
    seen_episodes: set[str] = set()
    for group in groups:
        if group.group_id in seen:
            raise ValueError(f"duplicate rollout group ID: {group.group_id}")
        seen.add(group.group_id)
        episode_ids = {episode.episode_id for episode in group.episodes}
        overlap = seen_episodes.intersection(episode_ids)
        if overlap:
            raise ValueError(f"duplicate episode ID across rollout groups: {min(overlap)}")
        seen_episodes.update(episode_ids)
        revisions = {episode.policy_revision for episode in group.episodes}
        if revisions != {config.expected_policy_revision}:
            raise ValueError(
                f"group {group.group_id} policy revision does not match the configured revision"
            )
        policy_ids = {episode.policy_id for episode in group.episodes}
        if policy_ids != {config.model_id}:
            raise ValueError(f"group {group.group_id} policy model does not match the config")


def train(config: GRPOConfig) -> Path:
    prepare_output_directory(config.output_dir)
    input_evidence = {
        "rollout_groups": artifact_evidence(config.rollout_groups_file).model_dump(mode="json"),
        "initial_adapter": artifact_evidence(config.initial_adapter).model_dump(mode="json"),
    }
    partial_manifest_path = config.output_dir / "run-manifest.partial.json"
    manifest_base = {
        "kind": "interactive-grpo",
        "status": "running",
        "git": git_state(),
        "config": config.model_dump(mode="json"),
        "config_sha256": canonical_sha256(config),
        "inputs": input_evidence,
    }
    partial_manifest_path.write_text(
        json.dumps(manifest_base, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    groups = read_grpo_groups_jsonl(config.rollout_groups_file)
    _validate_groups(groups, config)
    adapter_sha256 = "sha256:" + directory_sha256(config.initial_adapter)
    behavior_adapters = {
        episode.policy_adapter_sha256 for group in groups for episode in group.episodes
    }
    if behavior_adapters != {adapter_sha256}:
        raise ValueError("initial_adapter content does not match the behavior-policy trajectories")
    peft, torch, transformers = _runtime()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    dtype = torch.bfloat16 if config.bf16 else torch.float16
    model_kwargs: dict[str, Any] = {
        "revision": config.model_revision,
        "torch_dtype": dtype,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    if config.load_in_4bit:
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_id,
        **model_kwargs,
    )
    if config.load_in_4bit:
        base_model = peft.prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    elif config.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False
    model = peft.PeftModel.from_pretrained(
        base_model,
        str(config.initial_adapter),
        adapter_name="policy",
        is_trainable=True,
    )
    model.load_adapter(
        str(config.initial_adapter),
        adapter_name="reference",
        is_trainable=False,
    )
    model.set_adapter("policy")
    model.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("the policy adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    nonzero_groups = sum(not group.zero_variance for group in groups)
    if nonzero_groups == 0:
        raise RuntimeError("all rollout groups have zero reward variance")
    updates_per_epoch = math.ceil(nonzero_groups / config.gradient_accumulation_steps)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(0.03 * updates_per_epoch * config.epochs)),
        num_training_steps=updates_per_epoch * config.epochs,
    )
    device = next(model.parameters()).device
    metrics_path = config.output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    accumulated = 0

    for epoch in range(config.epochs):
        order = list(groups)
        random.Random(config.seed + epoch).shuffle(order)
        for group in order:
            rewards = [episode.reward for episode in group.episodes]
            group_ids = [group.group_id] * len(group.episodes)
            advantages, zero_variance = grouped_standardized_advantages(rewards, group_ids)
            if zero_variance:
                _append_metric(
                    metrics_path,
                    {
                        "epoch": epoch,
                        "group_id": group.group_id,
                        "skipped_zero_variance": True,
                    },
                )
                continue
            group_loss_value = 0.0
            kls: list[float] = []
            ratios: list[float] = []
            for episode, advantage in zip(group.episodes, advantages, strict=True):
                episode_tokens = episode.generated_tokens
                episode_loss_value = 0.0
                episode_kl_sum = 0.0
                episode_ratio_sum = 0.0
                for trace in episode.traces:
                    (
                        token_losses,
                        observed_kl_sum,
                        observed_ratio_sum,
                        max_abs_log_ratio,
                    ) = _trace_loss(
                        torch=torch,
                        model=model,
                        trace=trace,
                        advantage=advantage,
                        clip_epsilon=config.clip_epsilon,
                        kl_beta=config.kl_beta,
                        device=device,
                    )
                    if not bool(torch.isfinite(token_losses).all().detach().cpu()):
                        raise RuntimeError("GRPO produced a non-finite token loss")
                    if global_update == 0:
                        if max_abs_log_ratio > config.initial_log_ratio_tolerance:
                            raise RuntimeError(
                                "initial policy/behavior log-prob ratio exceeds tolerance; "
                                "rollout and trainer model preparation are inconsistent"
                            )
                    scaled_loss = token_losses.sum() / (
                        episode_tokens * len(group.episodes) * config.gradient_accumulation_steps
                    )
                    scaled_loss.backward()
                    episode_loss_value += float(token_losses.detach().sum().cpu())
                    episode_kl_sum += observed_kl_sum
                    episode_ratio_sum += observed_ratio_sum
                    del token_losses, scaled_loss
                episode_loss_value /= episode_tokens
                group_loss_value += episode_loss_value / len(group.episodes)
                kls.append(episode_kl_sum / episode_tokens)
                ratios.append(episode_ratio_sum / episode_tokens)
            accumulated += 1
            stepped = accumulated == config.gradient_accumulation_steps
            if stepped:
                torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                global_update += 1
            _append_metric(
                metrics_path,
                {
                    "epoch": epoch,
                    "group_id": group.group_id,
                    "loss": group_loss_value,
                    "mean_reward": sum(rewards) / len(rewards),
                    "reward_min": min(rewards),
                    "reward_max": max(rewards),
                    "mean_kl": sum(kls) / len(kls),
                    "mean_ratio": sum(ratios) / len(ratios),
                    "optimizer_step": stepped,
                    "global_update": global_update,
                    "skipped_zero_variance": False,
                },
            )
        if accumulated:
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            global_update += 1
        checkpoint = config.output_dir / f"checkpoint-epoch-{epoch + 1}"
        model.save_pretrained(checkpoint, selected_adapters=["policy"])

    final_dir = config.output_dir / "adapter-final"
    model.save_pretrained(final_dir, selected_adapters=["policy"])
    final_adapter_dir = _locate_saved_adapter(final_dir, "policy")
    output_adapter_sha256 = "sha256:" + directory_sha256(final_adapter_dir)
    manifest = {
        **manifest_base,
        "status": "completed",
        "transformers_version": str(transformers.__version__),
        "peft_version": str(peft.__version__),
        "torch_version": str(torch.__version__),
        "groups": len(groups),
        "zero_variance_groups": sum(group.zero_variance for group in groups),
        "optimizer_updates": global_update,
        "initial_adapter_sha256": adapter_sha256,
        "output_adapter": str(final_adapter_dir),
        "output_adapter_sha256": output_adapter_sha256,
        "on_policy_constraint": "recollect rollouts before the next policy iteration",
    }
    (config.output_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    partial_manifest_path.unlink()
    return final_adapter_dir


def _locate_saved_adapter(root: Path, adapter_name: str) -> Path:
    nested = root / adapter_name
    if (nested / "adapter_config.json").is_file():
        return nested
    if (root / "adapter_config.json").is_file():
        return root
    raise RuntimeError(f"PEFT did not write a loadable adapter under {root}")


def _append_metric(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
        handle.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_toml_config(args.config, GRPOConfig, section="grpo")
    adapter_dir = train(config)
    print(adapter_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
