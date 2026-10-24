"""
Multi-Series DataFrame Forecasting using Google TimesFM
Author: MD. RaKib Hassan Dipu (https://github.com/rakibdipu)
"""

import pandas as pd
import numpy as np
import timesfm


def main():
    print("Synthesizing dummy retail multi-store sales data...")
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    data = []

    for store_id in ["store_A", "store_B"]:
        for prod_id in ["item_1", "item_2"]:
            trend = np.linspace(50, 120, 180)
            noise = np.random.normal(0, 5, 180)
            sales = trend + noise + (15 if store_id == "store_B" else 0)
            for d, s in zip(dates, sales):
                data.append(
                    {
                        "store_id": store_id,
                        "product_id": prod_id,
                        "date": d,
                        "sales": max(0.0, float(s)),
                    }
                )

    df = pd.DataFrame(data)
    print(f"Synthesized DataFrame shape: {df.shape}")

    print("Initializing TimesFM model...")
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend="cpu", horizon_len=14),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )

    print("Running forecast across multiple time-series groups...")
    forecast_df = tfm.forecast_on_df(
        inputs=df,
        id_columns=["store_id", "product_id"],
        target_column="sales",
        time_column="date",
        freq="D",
        num_jobs=-1,
    )

    print("\n--- Forecast Output ---")
    print(forecast_df.head(10))


if __name__ == "__main__":
    main()
