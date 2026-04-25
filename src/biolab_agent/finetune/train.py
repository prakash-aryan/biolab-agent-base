"""QLoRA fine-tuning of MedGemma-4B via Unsloth.

Reads ``data/finetune/train.jsonl``, trains a LoRA adapter, and writes it to
``$BIOLAB_ARTIFACT_DIR/lora-protocol``. Intended VRAM target: 8 GB @ seqlen 1024.

Invoke as::

    python -m biolab_agent.finetune.train                 # defaults
    python -m biolab_agent.finetune.train --smoke         # 5 steps, for CI
    python -m biolab_agent.finetune.train --config configs/finetune.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger("biolab.finetune")


@dataclass
class Config:
    # Unsloth's prequantized 4-bit mirror of google/medgemma-4b-it.
    # Open-access (no HF gating) so the training script runs without tokens.
    base_model: str = "unsloth/medgemma-4b-it-bnb-4bit"
    max_seq_length: int = 512
    train_path: Path = Path("data/finetune/train.jsonl")
    eval_path: Path | None = Path("data/finetune/eval.jsonl")
    instruction_key: str = "instruction"
    response_key: str = "output"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    epochs: int = 3
    per_device_batch_size: int = 1
    grad_accumulation_steps: int = 8
    learning_rate: float = 2.0e-4
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    load_in_4bit: bool = True
    seed: int = 20260424
    output_dir: Path = Path("artifacts/lora-protocol")
    run_name: str = "medgemma-4b-protocol-lora"

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        raw = yaml.safe_load(path.read_text())
        train = raw.get("train", {})
        lora = raw.get("lora", {})
        ds = raw.get("dataset", {})
        return cls(
            base_model=raw.get("base_model", cls.base_model),
            max_seq_length=int(raw.get("max_seq_length", cls.max_seq_length)),
            train_path=Path(ds.get("train_path", cls.train_path)),
            eval_path=Path(ds["eval_path"]) if ds.get("eval_path") else None,
            instruction_key=ds.get("instruction_key", "instruction"),
            response_key=ds.get("response_key", "output"),
            lora_r=int(lora.get("r", cls.lora_r)),
            lora_alpha=int(lora.get("alpha", cls.lora_alpha)),
            lora_dropout=float(lora.get("dropout", cls.lora_dropout)),
            lora_target_modules=list(lora.get("target_modules") or cls().lora_target_modules),
            epochs=int(train.get("epochs", cls.epochs)),
            per_device_batch_size=int(
                train.get("per_device_batch_size", cls.per_device_batch_size)
            ),
            grad_accumulation_steps=int(
                train.get("grad_accumulation_steps", cls.grad_accumulation_steps)
            ),
            learning_rate=float(train.get("learning_rate", cls.learning_rate)),
            warmup_ratio=float(train.get("warmup_ratio", cls.warmup_ratio)),
            lr_scheduler=str(train.get("lr_scheduler", cls.lr_scheduler)),
            weight_decay=float(train.get("weight_decay", cls.weight_decay)),
            gradient_checkpointing=bool(
                train.get("gradient_checkpointing", cls.gradient_checkpointing)
            ),
            load_in_4bit=bool(train.get("load_in_4bit", cls.load_in_4bit)),
            seed=int(train.get("seed", cls.seed)),
            output_dir=Path(raw.get("output", {}).get("adapter_dir", cls.output_dir)),
            run_name=str(raw.get("output", {}).get("run_name", cls.run_name)),
        )


def _format_example(instruction: str, response: str) -> str:
    # Gemma chat template. Unsloth formats the tokenizer, but flat SFT on the
    # conversational prompt works well for short instruction pairs.
    return (
        "<start_of_turn>user\n"
        f"{instruction.strip()}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{response.strip()}\n"
        "<end_of_turn>"
    )


def _load_jsonl(path: Path, instr_key: str, resp_key: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        instr = obj.get(instr_key, "")
        resp = obj.get(resp_key, "")
        if not instr or not resp:
            continue
        rows.append({"text": _format_example(instr, resp)})
    return rows


def run_training(cfg: Config, *, smoke: bool = False) -> Path:
    # Heavy ML stack imported lazily so `--help` works without it installed.
    import datasets as _datasets
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
    from trl import SFTConfig, SFTTrainer

    # Gemma's tokenizer contains a ConfigModuleInstance that `dill` can't
    # pickle, which crashes the default HF datasets fingerprinting path.
    _datasets.disable_caching()

    log.info("Loading %s", cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The unsloth bnb-4bit mirror is already quantized, so we don't re-wrap it
    # with BitsAndBytesConfig. torch_dtype/device placement is inferred.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    # Safe even when the base is already quantized  -  it just enables
    # input gradients for the k-bit weights so PEFT LoRAs can train.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
    )
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_rows = _load_jsonl(cfg.train_path, cfg.instruction_key, cfg.response_key)
    log.info("Loaded %d training rows from %s", len(train_rows), cfg.train_path)
    if smoke:
        train_rows = train_rows[: max(16, cfg.per_device_batch_size * cfg.grad_accumulation_steps)]

    # Pre-tokenize in-process so the SFTTrainer does not try to `dataset.map`
    # the tokenizer at fingerprinting time  -  Gemma's tokenizer carries a
    # non-picklable ConfigModuleInstance that crashes dill.
    def _tokenize(text: str) -> dict[str, list[int]]:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
            add_special_tokens=False,
        )
        # Gemma3 rejects training input without token_type_ids (used upstream
        # to distinguish text vs image tokens). All zeros = text-only.
        enc["token_type_ids"] = [0] * len(enc["input_ids"])
        enc["labels"] = list(enc["input_ids"])
        return enc

    train_ds = Dataset.from_list([_tokenize(r["text"]) for r in train_rows])

    eval_ds = None
    if cfg.eval_path and cfg.eval_path.exists():
        eval_rows = _load_jsonl(cfg.eval_path, cfg.instruction_key, cfg.response_key)
        if eval_rows:
            eval_ds = Dataset.from_list([_tokenize(r["text"]) for r in eval_rows])
            log.info("Eval set: %d rows from %s", len(eval_rows), cfg.eval_path)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # SFTConfig in TRL 1.x subsumes TrainingArguments AND the old SFTTrainer
    # positional kwargs (dataset_text_field, max_seq_length, …).
    sft_args = SFTConfig(
        output_dir=str(cfg.output_dir),
        run_name=cfg.run_name,
        num_train_epochs=cfg.epochs if not smoke else 1,
        max_steps=5 if smoke else -1,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        weight_decay=cfg.weight_decay,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch" if not smoke else "no",
        seed=cfg.seed,
        report_to=[],
        # Our dataset is already tokenized (input_ids/labels), so SFTTrainer
        # should NOT re-tokenize or pack it.
        max_length=cfg.max_seq_length,
        packing=False,
        remove_unused_columns=False,
    )

    # default_data_collator preserves any pre-tokenized columns the dataset
    # carries (notably `token_type_ids`, which Gemma3 requires during training).
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
        data_collator=default_data_collator,
    )

    log.info("Starting training (smoke=%s)", smoke)
    trainer.train()

    log.info("Saving adapter to %s", cfg.output_dir)
    # Save only the LoRA adapter. The tokenizer comes from the base model
    # at inference time, so writing a 33 MB tokenizer.json next to the
    # adapter just bloats the artifact.
    model.save_pretrained(str(cfg.output_dir))
    return cfg.output_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="YAML config path")
    p.add_argument("--smoke", action="store_true", help="5-step smoke run for CI")
    p.add_argument("--base-model", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.base_model:
        cfg.base_model = args.base_model
    if args.output_dir:
        cfg.output_dir = args.output_dir
    out = run_training(cfg, smoke=args.smoke)
    log.info("Done. Adapter at %s", out)


if __name__ == "__main__":
    main()
