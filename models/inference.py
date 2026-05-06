"""
inference.py — Run question-answering inference with a fine-tuned BERT model.

Usage (CLI)
-----------
    python inference.py \
        --model_dir ./bert-qa-finetuned \
        --question "Where is the Eiffel Tower?" \
        --context  "The Eiffel Tower is located in Paris, France."

Usage (API)
-----------
    from inference import QAInferenceEngine
    engine = QAInferenceEngine("./bert-qa-finetuned")
    result = engine.predict("Who built the Eiffel Tower?",
                            "The Eiffel Tower was designed by Gustave Eiffel.")
    print(result)  # {'answer': 'Gustave Eiffel', 'score': 0.98, 'start': ..., 'end': ...}
"""

import argparse
import logging
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


class QAInferenceEngine:
    """Thin wrapper around HF pipeline for extractive QA."""

    def __init__(self, model_dir: str):
        device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading model from '%s' (device=%d)...", model_dir, device)
        self._pipe = pipeline(
            "question-answering",
            model     = model_dir,
            tokenizer = model_dir,
            device    = device,
        )
        logger.info("Model ready.")

    def predict(self, question: str, context: str) -> dict:
        """
        Extract the best answer span.

        Returns
        -------
        dict with keys: answer, score, start, end
        """
        result = self._pipe(question=question, context=context)
        return {
            "answer": result["answer"],
            "score" : round(result["score"], 4),
            "start" : result["start"],
            "end"   : result["end"],
        }

    def predict_batch(self, pairs: list[dict]) -> list[dict]:
        """
        Run inference on a list of {"question": ..., "context": ...} dicts.
        Returns a list of prediction dicts in the same order.
        """
        results = self._pipe(pairs)
        return [
            {
                "answer": r["answer"],
                "score" : round(r["score"], 4),
                "start" : r["start"],
                "end"   : r["end"],
            }
            for r in results
        ]


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="BERT QA Inference")
    parser.add_argument("--model_dir", type=str, default="./bert-qa-finetuned",
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--question", type=str, required=True, help="Question string")
    parser.add_argument("--context",  type=str, required=True, help="Context passage")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args   = _parse_args()
    engine = QAInferenceEngine(args.model_dir)
    result = engine.predict(args.question, args.context)

    print(f"\nQuestion : {args.question}")
    print(f"Answer   : {result['answer']}")
    print(f"Score    : {result['score']:.4f}")
    print(f"Span     : [{result['start']}, {result['end']}]")


if __name__ == "__main__":
    main()
