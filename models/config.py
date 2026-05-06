"""
config.py — Central configuration for BERT QA fine-tuning.
Edit values here; all other modules import from this file.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    # ── Dataset source ─────────────────────────────────────────────────────────
    # 'hub'  → load from Hugging Face Hub
    # 'csv'  → load from local CSV files
    source: str = "hub"

    # Hub settings
    hub_dataset_name: str   = "squad"
    hub_dataset_config: str = None        # e.g. 'v2' for SQuAD 2.0

    # CSV settings
    # Required columns: id, context, question, answers
    # 'answers' must be JSON: {"text": ["Paris"], "answer_start": [42]}
    csv_train_path: str = "data/train.csv"
    csv_val_path: str   = "data/validation.csv"


@dataclass
class ModelConfig:
    model_name: str  = "bert-base-uncased"
    output_dir: Path = Path("./bert-qa-finetuned")

    # Tokenisation
    max_length:  int = 384   # Max tokens per sample
    doc_stride:  int = 128   # Overlap when context is split into windows

    # Answer extraction (inference)
    n_best:            int = 20
    max_answer_length: int = 30


@dataclass
class TrainConfig:
    num_epochs:              int   = 3
    batch_size:              int   = 16
    learning_rate:           float = 2e-5
    weight_decay:            float = 0.01
    warmup_ratio:            float = 0.1
    metric_for_best_model:   str   = "f1"
    logging_steps:           int   = 100
    seed:                    int   = 42
    fp16:                    bool  = True   # Auto-disabled on CPU in trainer.py


@dataclass
class Config:
    data:  DataConfig  = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Singleton used by all modules
cfg = Config()
