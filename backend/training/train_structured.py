"""Fine-tune DistilGPT-2 on the Phase 2 atomic route-token objective."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
from collections import Counter
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
import transformers

from backend.climb_path.tokens import encode_route, special_token_config


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records[:max_samples] if max_samples else records


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StructuredRouteDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]], tokenizer, max_length: int) -> None:
        self.examples: list[dict[str, list[int]]] = []
        for record in records:
            encoding = encode_route(
                record,
                tokenizer.convert_tokens_to_ids,
                max_length=max_length,
            )
            self.examples.append(
                {
                    "input_ids": list(encoding.input_ids),
                    "attention_mask": list(encoding.attention_mask),
                    "labels": list(encoding.labels),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class CompletionOnlyCollator:
    """Dynamically pad examples while keeping prompt/padding loss masked."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: Sequence[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(example["input_ids"]) for example in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for example in examples:
            padding = max_length - len(example["input_ids"])
            batch["input_ids"].append(
                example["input_ids"] + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(example["attention_mask"] + [0] * padding)
            batch["labels"].append(example["labels"] + [-100] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


class FiniteLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and not math.isfinite(float(logs["loss"])):
            control.should_training_stop = True
            raise FloatingPointError(f"Non-finite training loss: {logs['loss']}")
        return control


def grade_sampling_weights(
    records: Sequence[dict[str, Any]], strength: float
) -> list[float] | None:
    """Return inverse-frequency weights without changing the epoch length.

    ``strength=0`` preserves ordinary shuffled sampling, ``0.5`` applies a
    square-root correction, and ``1`` gives every grade equal expected mass.
    """

    if not 0.0 <= strength <= 1.0:
        raise ValueError("grade sampling strength must be between 0 and 1")
    if strength == 0.0:
        return None
    counts = Counter(str(record["grade"]) for record in records)
    return [counts[str(record["grade"])] ** -strength for record in records]


class GradeBalancedTrainer(Trainer):
    """Trainer with an optional deterministic weighted training sampler."""

    def __init__(self, *args, sampling_weights: Sequence[float] | None = None, **kwargs):
        self.sampling_weights = (
            torch.as_tensor(sampling_weights, dtype=torch.double)
            if sampling_weights is not None
            else None
        )
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, train_dataset=None):
        if self.sampling_weights is None:
            return super()._get_train_sampler(train_dataset)
        generator = torch.Generator()
        generator.manual_seed(self.args.data_seed)
        return WeightedRandomSampler(
            self.sampling_weights,
            num_samples=len(self.sampling_weights),
            replacement=True,
            generator=generator,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--board-holds", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-name", default="distilbert/distilgpt2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--grade-sampling-strength",
        type=float,
        default=0.0,
        help="Inverse grade-frequency exponent: 0=off, 0.5=sqrt, 1=uniform grades.",
    )
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    set_seed(args.seed)

    board_catalog = json.loads(args.board_holds.read_text(encoding="utf-8"))
    board_holds = board_catalog["holds"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    added_tokens = tokenizer.add_special_tokens(special_token_config(board_holds))

    train_records = read_jsonl(args.train, args.max_train_samples)
    validation_records = read_jsonl(args.validation, args.max_validation_samples)
    sampling_weights = grade_sampling_weights(
        train_records, args.grade_sampling_strength
    )
    train_dataset = StructuredRouteDataset(train_records, tokenizer, args.max_length)
    validation_dataset = StructuredRouteDataset(
        validation_records, tokenizer, args.max_length
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    total_training_steps = math.ceil(steps_per_epoch * args.num_epochs)
    warmup_steps = round(total_training_steps * args.warmup_ratio)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to=["tensorboard"],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = GradeBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
        callbacks=[FiniteLossCallback()],
        sampling_weights=sampling_weights,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    evaluation = trainer.evaluate()
    model.config.use_cache = True
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = {
        "train_loss": float(train_result.metrics["train_loss"]),
        "validation_loss": float(evaluation["eval_loss"]),
        "train_runtime_seconds": float(train_result.metrics["train_runtime"]),
    }
    run_manifest = {
        "objective": "completion-only atomic MoonBoard hold generation",
        "base_model": args.model_name,
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "train_file_sha256": sha256_file(args.train),
        "validation_file_sha256": sha256_file(args.validation),
        "train_rows": len(train_records),
        "validation_rows": len(validation_records),
        "added_special_tokens": added_tokens,
        "tokenizer_size": len(tokenizer),
        "hyperparameters": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "grade_sampling_strength": args.grade_sampling_strength,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "metrics": metrics,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
