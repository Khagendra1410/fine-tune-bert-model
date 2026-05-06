"""
data_loader.py — Load QA datasets from the Hugging Face Hub or local CSV files.
"""

import json
import logging

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from config import cfg

logger = logging.getLogger(__name__)


def _parse_csv(path: str) -> Dataset:
    """
    Read a CSV file and return a HF Dataset.

    Required columns
    ----------------
    id       : unique string identifier
    context  : passage that contains the answer
    question : question string
    answers  : JSON string  →  {"text": ["answer"], "answer_start": [42]}
    """
    df = pd.read_csv(path)
    required = {"id", "context", "question", "answers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{path}' is missing required column(s): {missing}\n"
            "Expected: id, context, question, answers"
        )

    df["answers"] = df["answers"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    logger.info("Loaded %d rows from '%s'", len(df), path)
    return Dataset.from_pandas(df[["id", "context", "question", "answers"]])


def load_raw_datasets() -> DatasetDict:
    """
    Return a DatasetDict with 'train' and 'validation' splits.
    Source is controlled by cfg.data.source ('hub' | 'csv').
    """
    source = cfg.data.source.lower()

    if source == "hub":
        logger.info(
            "Loading '%s' (config=%s) from Hugging Face Hub...",
            cfg.data.hub_dataset_name,
            cfg.data.hub_dataset_config,
        )
        datasets = load_dataset(
            cfg.data.hub_dataset_name,
            cfg.data.hub_dataset_config,
        )

    elif source == "csv":
        logger.info(
            "Loading CSV dataset: train='%s'  val='%s'",
            cfg.data.csv_train_path,
            cfg.data.csv_val_path,
        )
        datasets = DatasetDict({
            "train":      _parse_csv(cfg.data.csv_train_path),
            "validation": _parse_csv(cfg.data.csv_val_path),
        })

    else:
        raise ValueError(
            f"cfg.data.source must be 'hub' or 'csv', got '{source}'."
        )

    logger.info(
        "Dataset loaded — train: %d  |  validation: %d",
        datasets["train"].num_rows,
        datasets["validation"].num_rows,
    )
    return datasets
