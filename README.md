# LegalMind — Legal Compliance Violation Classifier

> End-to-end LLM pipeline built from scratch: custom BPE tokenizer → 15M GPT pretrained on SEBI & GDPR corpora → fine-tuned classifier → FastAPI deployment with drift monitoring.

```
F1-score: 0.87 (violation class) | P95 latency: ~340ms CPU | RAM: <4 GB during inference
```


---
---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 6 GB | 8 GB |
| CPU | Any x86-64 | AVX-512 BF16 support |
| Python | 3.10+ | 3.11 |
| Storage | 2 GB | 5 GB |
| GPU | Not required | Optional (CUDA/MPS) |

> Tested on 8 GB RAM CPU-only. Pretraining takes ~4-8 hours on a modern laptop CPU.

---



## Model Architecture

```
Input text
    │
    ▼
BPE Tokenizer (vocab=8000, trained on SEBI+GDPR)
    │
    ▼
Token Embedding (8000 × 512) ──┐
                                + → Dropout
Positional Embedding (256×512) ┘
    │
    ▼
┌─────────────────────────────────┐
│  TransformerBlock × 6           │
│                                 │
│  RMSNorm                        │
│  GroupedQueryAttention          │
│    Q heads: 8                   │
│    KV heads: 2  (4:1 ratio)     │
│    d_head: 64                   │
│  RMSNorm                        │
│  FeedForward (512→1024→512)     │
└─────────────────────────────────┘
    │
    ▼
RMSNorm
    │
    ├──[pretrain]──► LM Head (512→8000) — next-token prediction
    │
    └──[finetune]──► Mean Pool → Linear(512→256) → GELU → Linear(256→2)
                                                            │
                                                   [compliant, violation]

Total parameters: ~14.5M
```

### Why GQA?

Standard Multi-Head Attention keeps 8 separate Key and Value projections.
Grouped Query Attention (used in LLaMA 2, Mistral) shares KV across query groups:

```
MHA:  8 Q heads + 8 K heads + 8 V heads  → full KV memory
GQA:  8 Q heads + 2 K heads + 2 V heads  → 4× less KV memory
```

On CPU inference with 8 GB RAM, this matters.

---

## Training Details

### Pretraining

| Hyperparameter | Value | Why |
|---|---|---|
| batch_size | 8 | RAM budget |
| seq_len | 256 | Covers most legal clauses |
| learning_rate | 3e-4 | Standard for small GPT |
| LR schedule | Cosine + warmup | Stable training |
| grad_clip | 1.0 | Prevent gradient explosion |
| gradient_checkpointing | True | Halves activation memory |
| bf16 autocast | Auto-detected | ~1.5× faster on modern CPU |
| max_steps | 20,000 | ~4-8h on CPU |

### Fine-tuning (2-phase)

**Phase 1 (epochs 1–2):** Freeze backbone → train classification head only.
Converges quickly, establishes reasonable decision boundary.

**Phase 2 (epoch 3+):** Unfreeze all → full fine-tuning at lower LR.
Adapts backbone representations to classification task.

| Hyperparameter | Value |
|---|---|
| batch_size | 16 |
| learning_rate | 1e-4 |
| class_weights | Inverse frequency |
| early_stopping | patience=3 |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Single text classification |
| POST | `/predict/batch` | Batch classification (up to 64) |
| GET | `/health` | Liveness probe |
| GET | `/drift` | Drift monitor status + KL stats |
| POST | `/drift/reset` | Reset drift reference window |
| GET | `/metrics` | P50/P95/P99 latency + violation rate |

Full interactive docs: `http://localhost:8000/docs`

---

## Drift Monitoring

The API tracks input distribution shift using KL divergence.

**How it works:**
1. First 50 requests build a reference token distribution
2. Every request: compute KL(live_window || reference)
3. If KL > 0.1 (configurable): `drift_alert: true` in response

**When to act on drift alerts:**
- Consistent drift → your users are sending different types of text than training data
- Re-label and fine-tune on new data
- Call `POST /drift/reset` after redeployment

---

## Labeled Data Format

```json
[
  {
    "text": "The broker executed trades using material non-public information...",
    "label": 1
  },
  {
    "text": "All disclosures were filed within the prescribed timelines...",
    "label": 0
  }
]
```

**Labels:** `0 = compliant`, `1 = violation`

**Recommended dataset sizes:**
- Minimum: 200 samples (100 per class)
- Good: 500 samples
- Excellent: 2000+ samples with real SEBI enforcement order text

---

## Memory Usage

| Stage | Peak RAM |
|-------|----------|
| Tokenizer training | ~500 MB |
| GPT pretraining (batch=8) | ~3.5 GB |
| GPT fine-tuning (batch=16) | ~2.5 GB |
| Inference (bf16) | ~350 MB |
| Inference (float32) | ~650 MB |

---

## Configuration

All hyperparameters are in `config.py`. Key settings:

```python
# config.py

ModelConfig:
  vocab_size     = 8000
  context_length = 256
  n_layers       = 6
  n_heads        = 8
  n_kv_heads     = 2    # GQA ratio
  d_model        = 512
  d_ff           = 1024

PretrainConfig:
  batch_size              = 8    # lower = less RAM
  gradient_checkpointing  = True # always True on 8GB
  max_steps               = 20000

APIConfig:
  use_bf16           = True
  drift_window       = 500
  drift_kl_threshold = 0.1
```

---

## Troubleshooting

**`RuntimeError: out of memory`**
→ Reduce `batch_size` in `config.py` (try 4 or even 2 for pretraining)
→ Ensure `gradient_checkpointing = True`
→ Close other applications before training

**`ModuleNotFoundError`**
→ Make sure you activated the virtualenv: `source venv/bin/activate`

**`FileNotFoundError: tokenizer.json`**
→ Run `python scripts/train_tokenizer.py` first

**`FileNotFoundError: checkpoints/pretrain/best.pt`**
→ Run `python scripts/run_pretrain.py` before fine-tuning

**SEBI scraper returns 0 documents**
→ SEBI website structure may have changed. Use `--rule-based` data generation
→ Or manually place text files in `data/raw/sebi/`

**Low F1 score (< 0.7)**
→ Not enough labeled data — generate more: `python scripts/generate_synthetic_data.py --n 1000`
→ Check class balance in your `labeled.json`
→ Try more fine-tuning epochs (increase `max_epochs` in config)

---

## Full Pipeline (one command per step)

```bash
pip install -r requirements.txt
pytest tests/ -v
python scripts/train_tokenizer.py --scrape
python scripts/generate_synthetic_data.py --n 500 --rule-based
python scripts/run_pretrain.py
python scripts/run_finetune.py --data data/finetune/labeled.json
python scripts/evaluate.py --data data/finetune/labeled.json
python scripts/serve.py
```

---

## License

MIT License — free to use, modify, and distribute.

---

*Built from scratch: no HuggingFace Transformers, no pretrained weights. Every component — tokenizer, attention, training loop, deployment — written in plain PyTorch.*
