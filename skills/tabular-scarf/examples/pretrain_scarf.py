"""
Pretrain SCARF (Self-Supervised Contrastive Learning) on Tabular Data
Author: MD. RaKib Hassan Dipu (https://github.com/rakibdipu)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pytorch_scarf.model import SCARF
from pytorch_scarf.loss import NTXent


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Generate synthetic tabular dataset (e.g. 2000 samples, 20 features)
    print("Generating synthetic tabular features...")
    np.random.seed(42)
    X = np.random.randn(2000, 20)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to PyTorch TensorDataset
    tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    input_dim = X.shape[1]
    emb_dim = 64

    # Instantiate SCARF model
    print(f"Instantiating SCARF (input_dim={input_dim}, emb_dim={emb_dim})...")
    model = SCARF(
        input_dim=input_dim,
        emb_dim=emb_dim,
        encoder_depth=4,
        head_depth=2,
        corruption_rate=0.6,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = NTXent(temperature=0.07)

    print("Beginning contrastive pre-training...")
    model.train()
    for epoch in range(1, 21):
        running_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Pass through SCARF (computes anchor representation & corrupted view)
            z_anchor, z_corrupted = model(batch)
            loss = criterion(z_anchor, z_corrupted)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            avg_loss = running_loss / len(loader)
            print(f"Epoch [{epoch:2d}/20] - NT-Xent Loss: {avg_loss:.4f}")

    # Save pretrained weights
    torch.save(model.state_dict(), "scarf_pretrained.pt")
    print("Model weights successfully saved to 'scarf_pretrained.pt'!")


if __name__ == "__main__":
    main()
