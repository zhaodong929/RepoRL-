"""QLoRA SFT entry point with assistant-only loss masking."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
from pathlib import Path
from typing import Any

from reporl.agent.hf_policy import directory_sha256
from reporl.agent.models import ChatMessage
from reporl.tasks.canonical import canonical_sha256
from reporl.training.config import SFTConfig, load_toml_config
from reporl.training.provenance import (
    artifact_evidence,
    git_state,
    prepare_output_directory,
)
from reporl.training.records import SFTRecord, read_sft_jsonl


class SFTTokenizationError(ValueError):
    pass


def _wire_message(message: ChatMessage) -> dict[str, str]:
    if message.role == "tool":
        return {
            "role": "user",
            "content": f"TOOL_OBSERVATION\n{message.content}",
        }
    return {"role": message.role, "content": message.content}


def _token_ids(tokenizer: Any, messages: list[dict[str, str]], *, generation: bool) -> list[int]:
    values = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation,
    )
    return [int(value) for value in values]


def tokenize_sft_record(
    tokenizer: Any,
    record: SFTRecord,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Mask every non-assistant token using the model's exact chat template."""

    wire_messages: list[dict[str, str]] = []
    spans: list[tuple[int, int]] = []
    rendered_prefixes: list[list[int]] = []
    for message in record.messages:
        wire = _wire_message(message)
        if message.role == "assistant":
            prompt_ids = _token_ids(tokenizer, wire_messages, generation=True)
            through_ids = _token_ids(
                tokenizer,
                [*wire_messages, wire],
                generation=False,
            )
            if through_ids[: len(prompt_ids)] != prompt_ids:
                raise SFTTokenizationError(
                    "chat template is not prefix-stable around assistant generation"
                )
            spans.append((len(prompt_ids), len(through_ids)))
            rendered_prefixes.append(through_ids)
        wire_messages.append(wire)

    input_ids = _token_ids(tokenizer, wire_messages, generation=False)
    if len(input_ids) > max_length:
        raise SFTTokenizationError(
            f"record {record.record_id} has {len(input_ids)} tokens, limit is {max_length}"
        )
    for prefix in rendered_prefixes:
        if input_ids[: len(prefix)] != prefix:
            raise SFTTokenizationError("chat template changed an earlier message prefix")
    labels = [-100] * len(input_ids)
    for start, end in spans:
        labels[start:end] = input_ids[start:end]
    if not any(label != -100 for label in labels):
        raise SFTTokenizationError(f"record {record.record_id} has no assistant tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _load_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        datasets = importlib.import_module("datasets")
        peft = importlib.import_module("peft")
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as error:
        raise RuntimeError("SFT requires RepoRL's 'training' optional dependencies") from error
    return datasets, peft, torch, transformers


def train(config: SFTConfig) -> Path:
    prepare_output_directory(
        config.output_dir,
        allow_nonempty=config.resume_from_checkpoint is not None,
    )
    input_evidence = {
        "train": artifact_evidence(config.train_file).model_dump(mode="json"),
        "eval": (
            artifact_evidence(config.eval_file).model_dump(mode="json")
            if config.eval_file is not None
            else None
        ),
        "resume_checkpoint": (
            artifact_evidence(config.resume_from_checkpoint).model_dump(mode="json")
            if config.resume_from_checkpoint is not None
            else None
        ),
    }
    partial_manifest_path = config.output_dir / "run-manifest.partial.json"
    manifest_base = {
        "kind": "sft",
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
    datasets, peft, torch, transformers = _load_runtime()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_dataset(path: Path) -> Any:
        accepted: list[dict[str, list[int]]] = []
        rejected: list[dict[str, str]] = []
        for record in read_sft_jsonl(path):
            try:
                accepted.append(
                    tokenize_sft_record(tokenizer, record, max_length=config.max_length)
                )
            except SFTTokenizationError as error:
                rejected.append({"record_id": record.record_id, "reason": str(error)})
        if not accepted:
            raise RuntimeError(f"no trainable records remain after tokenizing {path}")
        rejected_path = config.output_dir / f"rejected-{path.stem}.json"
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_path.write_text(json.dumps(rejected, indent=2), encoding="utf-8")
        return datasets.Dataset.from_list(accepted)

    train_dataset = build_dataset(config.train_file)
    eval_dataset = build_dataset(config.eval_file) if config.eval_file is not None else None
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
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if config.load_in_4bit:
        model = peft.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    lora_config = peft.LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, lora_config)
    arguments = transformers.TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_steps=config.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        bf16=config.bf16,
        fp16=not config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
    )
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    trainer = transformers.Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train(
        resume_from_checkpoint=(
            str(config.resume_from_checkpoint)
            if config.resume_from_checkpoint is not None
            else None
        )
    )
    adapter_dir = config.output_dir / "adapter-final"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    adapter_sha256 = "sha256:" + directory_sha256(adapter_dir)
    resolved_model_revision = str(
        getattr(model.config, "_commit_hash", None) or config.model_revision
    )
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
    resolved_tokenizer_revision = str(
        tokenizer_kwargs.get("_commit_hash") or resolved_model_revision
    )
    chat_template = str(getattr(tokenizer, "chat_template", None) or "")
    manifest = {
        **manifest_base,
        "status": "completed",
        "model_resolved_revision": resolved_model_revision,
        "tokenizer_resolved_revision": resolved_tokenizer_revision,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": "sha256:"
        + hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "transformers_version": str(transformers.__version__),
        "peft_version": str(peft.__version__),
        "torch_version": str(torch.__version__),
        "train_records": len(train_dataset),
        "eval_records": len(eval_dataset) if eval_dataset is not None else 0,
        "output_adapter": str(adapter_dir),
        "output_adapter_sha256": adapter_sha256,
    }
    (config.output_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    partial_manifest_path.unlink()
    return adapter_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_toml_config(args.config, SFTConfig, section="sft")
    adapter_dir = train(config)
    print(adapter_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
