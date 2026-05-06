# BERT Question Answering — Fine-Tuning

Fine-tune `bert-base-uncased` for extractive Question Answering using the Hugging Face ecosystem.  
Supports **SQuAD / any HF Hub QA dataset** as well as **custom local CSV files**.

---

## Project Structure

```
bert-qa-finetune/
├── config.py          # Central config — edit hyperparameters here
├── data_loader.py     # Load dataset from Hub or CSV
├── preprocess.py      # Tokenisation + answer span mapping
├── metrics.py         # EM & F1 post-processing
├── trainer.py         # HF Trainer setup & training loop
├── train.py           # ▶ Main entry point
├── evaluate.py        # Standalone evaluation script
├── inference.py       # CLI + Python API for inference
├── requirements.txt
└── finetune_bert_qa.ipynb  # Interactive Jupyter walkthrough
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-username/bert-qa-finetune.git
cd bert-qa-finetune
pip install -r requirements.txt

# 2. Train on SQuAD (default)
python train.py

# 3. Evaluate
python evaluate.py --model_dir ./bert-qa-finetuned

# 4. Inference
python inference.py \
  --model_dir ./bert-qa-finetuned \
  --question  "Where is the Eiffel Tower?" \
  --context   "The Eiffel Tower is located in Paris, France."
```

---

## Dataset Options

### Option A — Hugging Face Hub

Edit `config.py`:
```python
cfg.data.source          = "hub"
cfg.data.hub_dataset_name = "squad"  # or any HF QA dataset
```

Or pass via CLI:
```bash
python train.py --source hub --hub_dataset squad
```

### Option B — Custom CSV

Your CSV must have these columns:

| Column    | Type   | Example |
|-----------|--------|---------|
| `id`      | str    | `"q001"` |
| `context` | str    | `"The Eiffel Tower is in Paris..."` |
| `question`| str    | `"Where is the Eiffel Tower?"` |
| `answers` | JSON str | `{"text": ["Paris"], "answer_start": [27]}` |

```bash
python train.py --source csv --csv_train data/train.csv --csv_val data/val.csv
```

---

## Hyperparameter Overrides

All settings live in `config.py`. Common ones can also be passed as CLI flags:

```bash
python train.py \
  --epochs     5     \
  --lr         3e-5  \
  --batch_size 8     \
  --max_length 512   \
  --output_dir ./my-model
```

---

## Python Inference API

```python
from inference import QAInferenceEngine

engine = QAInferenceEngine("./bert-qa-finetuned")

result = engine.predict(
    question = "Who designed the Eiffel Tower?",
    context  = "The Eiffel Tower was designed by Gustave Eiffel and completed in 1889.",
)
print(result)
# {'answer': 'Gustave Eiffel', 'score': 0.9871, 'start': 34, 'end': 48}
```

---

## Key Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `model_name` | `bert-base-uncased` | Any HF QA-compatible model |
| `max_length` | 384 | Increase to 512 for longer passages |
| `doc_stride` | 128 | Sliding-window overlap |
| `num_epochs` | 3 | 2–4 is usually optimal |
| `learning_rate` | 2e-5 | Try 1e-5 to 5e-5 |
| `batch_size` | 16 | Reduce to 8 if OOM on GPU |

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- `transformers`, `datasets`, `evaluate`, `accelerate`

---

## License

MIT
