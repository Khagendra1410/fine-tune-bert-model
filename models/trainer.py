"""
trainer.py — Build and run the Hugging Face Trainer for BERT QA fine-tuning.
"""

import logging
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from config import cfg
from metrics import build_compute_metrics

logger = logging.getLogger(__name__)


def load_model() -> AutoModelForQuestionAnswering:
    """Load the base model and move it to the best available device."""
    model = AutoModelForQuestionAnswering.from_pretrained(cfg.model.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded: %s  |  total params: %s  |  trainable: %s  |  device: %s",
        cfg.model.model_name,
        f"{total:,}",
        f"{trainable:,}",
        device,
    )
    return model


def build_training_args() -> TrainingArguments:
    use_fp16 = cfg.train.fp16 and torch.cuda.is_available()
    return TrainingArguments(
        output_dir                  = str(cfg.model.output_dir),
        num_train_epochs            = cfg.train.num_epochs,
        per_device_train_batch_size = cfg.train.batch_size,
        per_device_eval_batch_size  = cfg.train.batch_size,
        learning_rate               = cfg.train.learning_rate,
        weight_decay                = cfg.train.weight_decay,
        warmup_ratio                = cfg.train.warmup_ratio,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = cfg.train.metric_for_best_model,
        greater_is_better           = True,
        logging_dir                 = str(cfg.model.output_dir / "logs"),
        logging_steps               = cfg.train.logging_steps,
        fp16                        = use_fp16,
        seed                        = cfg.train.seed,
        report_to                   = "none",
    )


def run_training(
    train_dataset: Dataset,
    validation_dataset_full: Dataset,
    raw_val_dataset: Dataset,
    tokenizer,
) -> Trainer:
    """
    Fine-tune the model and save the best checkpoint.

    Parameters
    ----------
    train_dataset            : tokenised training features
    validation_dataset_full  : tokenised val features (with offset_mapping / example_id)
    raw_val_dataset          : original (un-tokenised) validation examples
    tokenizer                : tokenizer used during preprocessing

    Returns
    -------
    trainer : fitted Trainer instance (best model already loaded)
    """
    model        = load_model()
    training_args = build_training_args()

    # Trainer needs a clean eval set (no offset_mapping / example_id columns)
    val_for_trainer = validation_dataset_full.remove_columns(
        ["example_id", "offset_mapping"]
    )

    compute_metrics = build_compute_metrics(
        raw_val_dataset, validation_dataset_full, tokenizer
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_for_trainer,
        tokenizer       = tokenizer,
        data_collator   = DefaultDataCollator(),
        compute_metrics = compute_metrics,
    )

    logger.info("Starting training (%d epochs)...", cfg.train.num_epochs)
    train_result = trainer.train()

    logger.info("Training complete. Saving model to '%s'...", cfg.model.output_dir)
    trainer.save_model(str(cfg.model.output_dir))
    tokenizer.save_pretrained(str(cfg.model.output_dir))

    logger.info("Train metrics: %s", train_result.metrics)
    return trainer
