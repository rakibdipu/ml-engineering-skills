---
name: timesfm-forecasting
description: >-
  Zero-shot and fine-tuned time-series forecasting using Google Research's
  TimesFM (Time Series Foundation Model). Use this skill when predicting future
  trends, sales demand, weather, energy, or financial metrics; handling
  univariate or multivariate time-series data with covariates; evaluating
  time-series foundation models without training from scratch; or integrating
  TimesFM into production via Google Cloud BigQuery AI.FORECAST.
---

# TimesFM — Time Series Foundation Model

> **Source:** [google-research/timesfm](https://github.com/google-research/timesfm)  
> **Model Cards:** [google/timesfm-1.0-200m-pytorch](https://huggingface.co/google/timesfm-1.0-200m-pytorch) · [google/timesfm-2.0-500m-pytorch](https://huggingface.co/google/timesfm-2.0-500m-pytorch)

Google Research's **TimesFM** is a pretrained decoder-only foundation model trained on **over 100 billion** real-world and synthetic time points. Like an LLM for numbers, it performs **zero-shot forecasting** across domains without task-specific retraining.

---

## 1. Installation & Setup

```bash
# Core package — choose the right extra for your backend
pip install "timesfm[torch]"     # PyTorch (CUDA) — recommended for most users
pip install "timesfm[jax]"       # JAX / TPU
pip install huggingface_hub      # Required to pull checkpoints from HF Hub
```

### Available Checkpoints

| Checkpoint | Params | Notes |
| :--- | :--- | :--- |
| `google/timesfm-1.0-200m-pytorch` | 200M | Production stable, fast, univariate |
| `google/timesfm-2.0-500m-pytorch` | 500M | Multivariate, covariate support, higher accuracy |

> **Do NOT hardcode `num_layers` or `model_dims` manually.** These hyper-parameters are baked into each checkpoint and are loaded automatically when you specify `huggingface_repo_id`. Overriding them will break the model.

---

## 2. Quickstart: Zero-Shot Forecasting

```python
import numpy as np
import timesfm

# Load model — hparams are inferred from the checkpoint automatically
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="gpu",       # "gpu", "cpu", or "tpu"
        horizon_len=24,      # Number of future steps to predict
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    ),
)

# Historical time-series data — variable length OK, but ≥128 steps gives best results
history_batch = [
    np.sin(np.linspace(0, 20, 256)) + np.random.normal(0, 0.05, 256),
    np.cos(np.linspace(0, 20, 256)) + np.random.normal(0, 0.05, 256),
]

# Frequency indicator:
# 0 = High frequency  (sub-daily: minutes, hours)
# 1 = Daily
# 2 = Low frequency   (weekly, monthly, quarterly, yearly)
frequencies = [0, 0]

# Returns point forecasts and full quantile distributions
point_forecast, quantile_forecast = tfm.forecast(
    inputs=history_batch,
    freq=frequencies,
)

# point_forecast.shape   → (batch_size, horizon_len)
# quantile_forecast.shape → (batch_size, horizon_len, n_quantiles)  ← uncertainty bands
print("Point Forecast (next 24 steps):", point_forecast[0])
```

---

## 3. Forecasting on Pandas DataFrames (Multi-Series)

Real-world use case: forecasting multiple stores / products / sensors simultaneously.

```python
import pandas as pd
import timesfm

df = pd.read_csv("sales.csv")
# Expected columns: ["store_id", "product_id", "date", "sales"]
# Timestamps must be regularly spaced. Interpolate NaNs before calling.

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(backend="cpu", horizon_len=14),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    ),
)

forecast_df = tfm.forecast_on_df(
    inputs=df,
    id_columns=["store_id", "product_id"],   # Columns that uniquely identify a series
    target_column="sales",                    # The variable to predict
    time_column="date",                       # Timestamp column
    freq="D",            # Pandas offset alias: 'D'=daily, 'H'=hourly, 'W'=weekly, 'M'=monthly
    num_jobs=-1,         # Parallel jobs (-1 = use all CPU cores)
)

print(forecast_df[["store_id", "product_id", "date", "sales_mean",
                    "sales_q10", "sales_q90"]].head())
```

---

## 4. Covariate Support (TimesFM 2.0+)

TimesFM 2.0 supports **known future covariates** (e.g., holiday flags, price, temperature) and **past covariates** (e.g., lagged promotions) to improve accuracy.

```python
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(backend="gpu", horizon_len=30),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
    ),
)

forecast_df = tfm.forecast_on_df(
    inputs=df,
    id_columns=["series_id"],
    target_column="demand",
    time_column="date",
    freq="D",
    # Future covariates: known for the full forecast horizon
    future_covariates=["is_holiday", "price_discount"],
    # Past covariates: only known historically, not in the future
    past_covariates=["promo_spend"],
)
```

---

## 5. Fine-Tuning on Your Domain Data

For domain-specific accuracy gains (medical, financial, industrial IoT), fine-tune the pretrained model:

```python
import torch
from timesfm import TimesFm, TimesFmHparams, TimesFmCheckpoint

tfm = TimesFm(
    hparams=TimesFmHparams(backend="gpu", horizon_len=24),
    checkpoint=TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    ),
)

# Fine-tuning is done by calling .finetune() with your labeled dataset
# (see official TimesFM fine-tuning notebook in the repo for full DataLoader setup)
tfm.finetune(
    train_data=train_dataset,      # torch.utils.data.Dataset returning (context, horizon)
    eval_data=eval_dataset,
    learning_rate=1e-4,
    num_epochs=10,
    batch_size=64,
)

# Save the fine-tuned weights
tfm.save_checkpoint("./my-timesfm-finetuned")

# Later: reload
tfm_loaded = TimesFm(
    hparams=TimesFmHparams(backend="gpu", horizon_len=24),
    checkpoint=TimesFmCheckpoint(local_path="./my-timesfm-finetuned"),
)
```

---

## 6. Evaluation — Benchmark vs. Baselines

Always evaluate using **MAE**, **MASE**, and **sMAPE** against simple baselines:

```python
import numpy as np
from sklearn.metrics import mean_absolute_error

def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8))

# Ground truth vs TimesFM
mae  = mean_absolute_error(y_true, point_forecast[0])
smap = smape(y_true, point_forecast[0])
print(f"MAE: {mae:.4f}  |  sMAPE: {smap:.2f}%")
```

---

## 7. Google Cloud BigQuery Integration

For enterprise-scale forecasting with no GPU infrastructure required:

```sql
-- Uses TimesFM natively via BigQuery ML
SELECT *
FROM AI.FORECAST(
  MODEL `my_project.my_dataset.my_timesfm_model`,
  TABLE `my_project.my_dataset.sales_data`,
  STRUCT(
    30  AS horizon,
    0.9 AS confidence_level
  )
);
```

---

## 8. Key Parameters Reference

| Parameter | Recommended | Description |
| :--- | :--- | :--- |
| `horizon_len` | 1–512 | Steps into the future to predict |
| `freq` | 0, 1, 2 | 0=sub-daily, 1=daily, 2=weekly+ |
| `backend` | `"gpu"` / `"cpu"` | Hardware backend |
| Context length | ≥128 (ideal 512) | More history → better forecasts |

### Pitfalls to Avoid
1. **Irregular timestamps:** Resample or forward-fill before calling TimesFM.
2. **Overriding model architecture hparams:** Let the checkpoint define these.
3. **Mixing frequencies in one batch:** Group by freq before batching.
4. **Ignoring quantile outputs:** Always surface `q10`/`q90` for business decisions requiring uncertainty estimates (inventory safety stock, risk bounds).
