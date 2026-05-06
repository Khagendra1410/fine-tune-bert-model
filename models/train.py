"""
train.py — Main entry point for BERT QA fine-tuning.

Usage
-----
    # Default (SQuAD from Hub, all settings from config.py)
    python train.py

    # Override dataset source via CLI
    python train.py --source hub --hub_dataset squad
    python train.py --source csv --csv_train data/train.csv --csv_val data/val.csv

    # Quickly override common hyperparameters
    python train.py --epochs 5 --lr 3e-5 --batch_size 8
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from config import cfg
from data_loader import load_raw_datasets
from preprocess import get_tokenizer, tokenize_datasets
from trainer import run_training

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt= "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Question Answering")

    # Dataset
    parser.add_argument("--source",      type=str, help="'hub' or 'csv'")
    parser.add_argument("--hub_dataset", type=str, help="HF Hub dataset name")
    parser.add_argument("--csv_train",   type=str, help="Path to train CSV")
    parser.add_argument("--csv_val",     type=str, help="Path to validation CSV")

    # Model / tokeniser
    parser.add_argument("--model_name",  type=str, help="HF model checkpoint name")
    parser.add_argument("--output_dir",  type=str, help="Where to save the model")
    parser.add_argument("--max_length",  type=int, help="Max token length")
    parser.add_argument("--doc_stride",  type=int, help="Sliding-window overlap")

    # Training
    parser.add_argument("--epochs",      type=int,   help="Number of training epochs")
    parser.add_argument("--batch_size",  type=int,   help="Per-device batch size")
    parser.add_argument("--lr",          type=float, help="Learning rate")
    parser.add_argument("--seed",        type=int,   help="Random seed")

    return parser.parse_args()


def apply_cli_overrides(args):
    """Overwrite cfg values with any CLI arguments that were provided."""
    if args.source:      cfg.data.source           = args.source
    if args.hub_dataset: cfg.data.hub_dataset_name = args.hub_dataset
    if args.csv_train:   cfg.data.csv_train_path   = args.csv_train
    if args.csv_val:     cfg.data.csv_val_path      = args.csv_val

    if args.model_name:  cfg.model.model_name = args.model_name
    if args.output_dir:  cfg.model.output_dir = Path(args.output_dir)
    if args.max_length:  cfg.model.max_length = args.max_length
    if args.doc_stride:  cfg.model.doc_stride = args.doc_stride

    if args.epochs:      cfg.train.num_epochs    = args.epochs
    if args.batch_size:  cfg.train.batch_size    = args.batch_size
    if args.lr:          cfg.train.learning_rate = args.lr
    if args.seed:        cfg.train.seed          = args.seed


def main():
    args = parse_args()
    apply_cli_overrides(args)

    logger.info("=== BERT QA Fine-Tuning ===")
    logger.info("Model      : %s", cfg.model.model_name)
    logger.info("Source     : %s", cfg.data.source)
    logger.info("Output dir : %s", cfg.model.output_dir)

    # 1. Load raw dataset
    raw_datasets = load_raw_datasets()

    # 2. Tokenise
    tokenizer = get_tokenizer()
    train_dataset, val_dataset_full = tokenize_datasets(raw_datasets, tokenizer)

    # 3. Train
    trainer = run_training(
        train_dataset        = train_dataset,
        validation_dataset_full = val_dataset_full,
        raw_val_dataset      = raw_datasets["validation"],
        tokenizer            = tokenizer,
    )

    # 4. Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info("Final eval results: %s", eval_results)

    # 5. Save results JSON
    results_path = cfg.model.output_dir / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"eval": eval_results}, f, indent=2)
    logger.info("Results saved to '%s'", results_path)
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
