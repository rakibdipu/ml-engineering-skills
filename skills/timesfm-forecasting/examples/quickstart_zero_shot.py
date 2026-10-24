"""
Quickstart Zero-Shot Time Series Forecasting using Google TimesFM
Author: MD. RaKib Hassan Dipu (https://github.com/rakibdipu)
"""

import numpy as np
import timesfm


def main():
    print("Loading Google TimesFM checkpoint (200M PyTorch)...")
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",  # Set to "gpu" if CUDA is available
            horizon_len=24,  # Forecast next 24 steps
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )

    print("Synthesizing sample historical time series...")
    t = np.linspace(0, 20, 256)
    history_batch = [
        np.sin(t) + np.random.normal(0, 0.05, 256),
        np.cos(t) + np.random.normal(0, 0.05, 256),
    ]

    # Frequency: 0 = High (< daily), 1 = Daily, 2 = Low (> daily)
    frequencies = [0, 0]

    print("Generating zero-shot forecast...")
    point_forecast, quantile_forecast = tfm.forecast(
        inputs=history_batch,
        freq=frequencies,
    )

    print("\n--- Results ---")
    print(f"Batch Size: {point_forecast.shape[0]}")
    print(f"Horizon Length: {point_forecast.shape[1]}")
    print(f"Series 1 point forecast (first 5 steps): {point_forecast[0][:5]}")
    print(f"Series 2 point forecast (first 5 steps): {point_forecast[1][:5]}")


if __name__ == "__main__":
    main()
