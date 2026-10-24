"""
Downstream Classification using Pretrained SCARF Embeddings + XGBoost
Author: MD. RaKib Hassan Dipu (https://github.com/rakibdipu)
"""

import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from pytorch_scarf.model import SCARF


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "scarf_pretrained.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Run pretrain_scarf.py first.")
        return

    # Simulate labeled tabular dataset
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    # Non-linear synthetic target based on feature interactions
    y = (X[:, 0] * X[:, 1] + X[:, 2] ** 2 > 0.5).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model architecture and load weights
    model = SCARF(
        input_dim=20,
        emb_dim=64,
        encoder_depth=4,
        head_depth=2,
        corruption_rate=0.6,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    def extract_features(data_matrix: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(data_matrix, dtype=torch.float32).to(device)
            # Use raw encoder representation (not the projection head)
            return model.encoder(t).cpu().numpy()

    print("Extracting dense representations from pre-trained SCARF encoder...")
    X_train_emb = extract_features(X_train)
    X_test_emb = extract_features(X_test)

    print("Training XGBoost Classifier on top of SCARF embeddings...")
    clf = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf.fit(X_train_emb, y_train)

    y_pred = clf.predict(X_test_emb)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nDownstream Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
