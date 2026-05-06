"""
evaluate.py — Evaluate a saved fine-tuned model on a QA dataset.

Usage
-----
    python evaluate.py --model_dir ./bert-qa-finetuned
    python evaluate.py --model_dir ./bert-qa-finetuned --source csv --csv_val data/val.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments, DefaultDataCollator

from config import cfg
from data_loader import load_raw_datasets
from preprocess import tokenize_datasets
from metrics import build_compute_metrics, postprocess_qa_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned BERT QA model")
    parser.add_argument("--model_dir", type=str, default="./bert-qa-finetuned",
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--source",    type=str, help="'hub' or 'csv'")
    parser.add_argument("--csv_val",   type=str, help="Path to validation CSV")
    parser.add_argument("--output",    type=str, default="eval_results.json",
                        help="Where to write evaluation JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source:  cfg.data.source       = args.source
    if args.csv_val: cfg.data.csv_val_path = args.csv_val

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error("Model directory '%s' does not exist.", model_dir)
        sys.exit(1)

    logger.info("Loading tokenizer and model from '%s'...", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    logger.info("Loading and tokenising dataset...")
    raw_datasets = load_raw_datasets()
    _, val_dataset_full = tokenize_datasets(raw_datasets, tokenizer)

    val_for_trainer = val_dataset_full.remove_columns(["example_id", "offset_mapping"])

    # Use a minimal Trainer just for prediction
    eval_args = TrainingArguments(
        output_dir            = str(model_dir),
        per_device_eval_batch_size = cfg.train.batch_size,
        report_to             = "none",
        no_cuda               = not torch.cuda.is_available(),
    )

    compute_metrics = build_compute_metrics(
        raw_datasets["validation"], val_dataset_full, tokenizer
    )

    trainer = Trainer(
        model           = model,
        args            = eval_args,
        eval_dataset    = val_for_trainer,
        tokenizer       = tokenizer,
        data_collator   = DefaultDataCollator(),
        compute_metrics = compute_metrics,
    )

    logger.info("Running evaluation...")
    results = trainer.evaluate()

    logger.info("─" * 45)
    for k, v in results.items():
        logger.info("  %-30s : %s", k, f"{v:.4f}" if isinstance(v, float) else v)
    logger.info("─" * 45)

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to '%s'", out_path)


if __name__ == "__main__":
    main()
