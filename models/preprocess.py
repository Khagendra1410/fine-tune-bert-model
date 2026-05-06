"""
preprocess.py — Tokenise QA examples and map answer character spans to token indices.

For long contexts the tokeniser uses a sliding-window approach (doc_stride) so
that every part of the passage appears in at least one feature.
"""

import logging
from datasets import DatasetDict
from transformers import AutoTokenizer

from config import cfg

logger = logging.getLogger(__name__)


def get_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    logger.info("Tokenizer loaded: %s  (vocab=%d)", cfg.model.model_name, tokenizer.vocab_size)
    return tokenizer


# ── Training features ─────────────────────────────────────────────────────────

def _make_train_fn(tokenizer: AutoTokenizer):
    """Return a batched map function for training examples."""

    max_length = cfg.model.max_length
    doc_stride = cfg.model.doc_stride

    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map     = inputs.pop("overflow_to_sample_mapping")
        answers        = examples["answers"]
        start_positions, end_positions = [], []

        for i, offset in enumerate(offset_mapping):
            sample_idx   = sample_map[i]
            answer       = answers[sample_idx]
            input_ids    = inputs["input_ids"][i]
            cls_index    = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = inputs.sequence_ids(i)

            # Find context token window
            ctx_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
            ctx_end   = len(sequence_ids) - 1
            while sequence_ids[ctx_end] != 1:
                ctx_end -= 1

            # No answer or answer outside this window → CLS
            if len(answer["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            char_start = answer["answer_start"][0]
            char_end   = char_start + len(answer["text"][0])

            if offset[ctx_start][0] > char_end or offset[ctx_end][1] < char_start:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            tok_start = ctx_start
            while tok_start <= ctx_end and offset[tok_start][0] <= char_start:
                tok_start += 1
            start_positions.append(tok_start - 1)

            tok_end = ctx_end
            while tok_end >= ctx_start and offset[tok_end][1] >= char_end:
                tok_end -= 1
            end_positions.append(tok_end + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"]   = end_positions
        return inputs

    return preprocess_training_examples


# ── Validation features ───────────────────────────────────────────────────────

def _make_val_fn(tokenizer: AutoTokenizer):
    """Return a batched map function for validation examples.

    Keeps offset_mapping and example_id so predictions can be mapped back
    to the original answer spans for scoring.
    """

    max_length = cfg.model.max_length
    doc_stride = cfg.model.doc_stride

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map  = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx   = sample_map[i]
            sequence_ids = inputs.sequence_ids(i)
            example_ids.append(examples["id"][sample_idx])
            inputs["offset_mapping"][i] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(inputs["offset_mapping"][i])
            ]

        inputs["example_id"] = example_ids
        return inputs

    return preprocess_validation_examples


# ── Public API ────────────────────────────────────────────────────────────────

def tokenize_datasets(
    raw_datasets: DatasetDict,
    tokenizer: AutoTokenizer,
) -> tuple[DatasetDict, DatasetDict]:
    """
    Tokenise train and validation splits.

    Returns
    -------
    train_dataset              : ready for Trainer (no extra columns)
    validation_dataset_full    : includes 'example_id' and 'offset_mapping'
                                 needed for post-processing predictions
    """
    train_fn = _make_train_fn(tokenizer)
    val_fn   = _make_val_fn(tokenizer)

    train_dataset = raw_datasets["train"].map(
        train_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenising train set",
    )

    validation_dataset_full = raw_datasets["validation"].map(
        val_fn,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        desc="Tokenising validation set",
    )

    logger.info(
        "Tokenisation done — train features: %d  |  val features: %d",
        train_dataset.num_rows,
        validation_dataset_full.num_rows,
    )
    return train_dataset, validation_dataset_full
