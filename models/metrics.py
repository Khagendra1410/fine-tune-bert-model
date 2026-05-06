"""
metrics.py — Exact Match & F1 scoring for extractive QA.

Post-processes raw start/end logits → answer strings, then scores
against ground-truth using the official SQuAD metric.
"""

import logging
import numpy as np
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer

from config import cfg

logger = logging.getLogger(__name__)

_squad_metric = evaluate.load("squad")


def postprocess_qa_predictions(
    examples: Dataset,
    features: Dataset,
    raw_predictions: tuple,
    tokenizer: AutoTokenizer,
    n_best: int = None,
    max_answer_length: int = None,
) -> dict[str, str]:
    """
    Convert raw start/end logits to the best answer string per example.

    Parameters
    ----------
    examples          : original (un-tokenised) validation examples
    features          : tokenised validation features (with offset_mapping)
    raw_predictions   : (start_logits, end_logits) arrays from the model
    tokenizer         : tokenizer used during preprocessing
    n_best            : how many top positions to consider (default from cfg)
    max_answer_length : maximum tokens in a valid answer span (default from cfg)

    Returns
    -------
    dict mapping example id → predicted answer string
    """
    n_best            = n_best or cfg.model.n_best
    max_answer_length = max_answer_length or cfg.model.max_answer_length

    all_start_logits, all_end_logits = raw_predictions

    # Build example_id → feature index list
    features_per_example: dict[str, list[int]] = {}
    for i, feat in enumerate(features):
        eid = feat["example_id"]
        features_per_example.setdefault(eid, []).append(i)

    predictions: dict[str, str] = {}

    for example in examples:
        eid     = example["id"]
        context = example["context"]
        min_null_score = None
        valid_answers  = []

        for feat_idx in features_per_example.get(eid, []):
            start_logits   = all_start_logits[feat_idx]
            end_logits     = all_end_logits[feat_idx]
            offset_mapping = features[feat_idx]["offset_mapping"]
            input_ids      = features[feat_idx]["input_ids"]
            cls_index      = input_ids.index(tokenizer.cls_token_id)

            null_score = float(start_logits[cls_index]) + float(end_logits[cls_index])
            if min_null_score is None or null_score < min_null_score:
                min_null_score = null_score

            start_indices = np.argsort(start_logits)[-n_best:][::-1].tolist()
            end_indices   = np.argsort(end_logits)[-n_best:][::-1].tolist()

            for si in start_indices:
                for ei in end_indices:
                    if (
                        offset_mapping[si] is None
                        or offset_mapping[ei] is None
                        or ei < si
                        or ei - si + 1 > max_answer_length
                    ):
                        continue
                    valid_answers.append({
                        "score": float(start_logits[si]) + float(end_logits[ei]),
                        "text" : context[offset_mapping[si][0]: offset_mapping[ei][1]],
                    })

        best = (
            max(valid_answers, key=lambda x: x["score"])
            if valid_answers
            else {"text": ""}
        )
        predictions[eid] = best["text"]

    return predictions


def build_compute_metrics(raw_val_dataset: Dataset, val_features: Dataset, tokenizer: AutoTokenizer):
    """
    Return a compute_metrics function compatible with HF Trainer.

    Closes over the raw validation examples and features so the Trainer
    can call it with only (EvalPrediction,).
    """

    def compute_metrics(eval_preds):
        preds = postprocess_qa_predictions(
            raw_val_dataset,
            val_features,
            eval_preds.predictions,
            tokenizer,
        )
        formatted_preds = [{"id": k, "prediction_text": v} for k, v in preds.items()]
        references = [
            {"id": ex["id"], "answers": ex["answers"]}
            for ex in raw_val_dataset
        ]
        result = _squad_metric.compute(predictions=formatted_preds, references=references)
        logger.info("Eval → EM: %.2f  F1: %.2f", result["exact_match"], result["f1"])
        return result

    return compute_metrics
