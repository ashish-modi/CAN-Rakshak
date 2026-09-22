

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ids.base import IDS


class EnhancedSelfAttention(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=24):
        super().__init__()
        self.query = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.key = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.value = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.output_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn_map = torch.sigmoid(Q * K)
        weighted_V = attn_map * V
        out = self.output_conv(weighted_V)
        return out.squeeze(1)


class MULSAMNet(nn.Module):
    def __init__(self, input_size=11, hidden_size=24, num_classes=2):
        super().__init__()
        self.attention = EnhancedSelfAttention(input_size, hidden_size)
        self.lstm_time = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_depth = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x_enhanced = self.attention(x)
        out_t, (h_t, _) = self.lstm_time(x_enhanced)
        out_d, (h_d, _) = self.lstm_depth(x_enhanced)
        combined = torch.cat((h_t[-1], h_d[-1]), dim=1)
        return self.fc(combined)


class MULSAM(IDS):
    """MULSAM Intrusion Detection System (raw CAN-ID bitstream, windowed)."""

    WINDOW_SIZE = 32
    HIDDEN_DIM = 24
    NUM_CLASSES = 2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MULSAMNet(
            hidden_size=self.HIDDEN_DIM, num_classes=self.NUM_CLASSES
        ).to(self.device)


    def train(self, train_dataset_dir=None, X_train=None, Y_train=None, cfg=None, **kwargs):
        cfg = cfg or {}
        csv_path = self._get_csv_path(cfg)
        print(f"MULSAM — loading training data from {csv_path}")
        X, y = self._load_windows(csv_path)
        print(f"  windows: {len(X)}  attack: {int(y.sum())}  benign: {int((y == 0).sum())}")

        train_ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        train_loader = DataLoader(train_ds, batch_size=self.BATCH_SIZE, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        epochs = cfg.get('epochs', 50)

        print(f"Training MULSAM on {self.device} ({epochs} epochs)")
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_b), y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1:3d}/{epochs}  loss={total_loss / len(train_loader):.4f}")

    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs):
        cfg = cfg or {}
        csv_path = self._get_csv_path(cfg)
        print(f"MULSAM — loading test data from {csv_path}")
        X, y = self._load_windows(csv_path)
        print(f"  windows: {len(X)}  attack: {int(y.sum())}  benign: {int((y == 0).sum())}")

        preds = self.predict(X)
        return preds, y

    def predict(self, X_test, **kwargs):
        self.model.eval()
        ds = TensorDataset(torch.from_numpy(np.asarray(X_test, dtype=np.float32)))
        loader = DataLoader(ds, batch_size=self.BATCH_SIZE)
        all_preds = []
        with torch.no_grad():
            for (X_b,) in loader:
                out = self.model(X_b.to(self.device))
                all_preds.extend(out.argmax(dim=1).cpu().numpy())
        return np.array(all_preds)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"  MULSAM model saved to {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"  MULSAM model loaded from {path}")

    def extract_features(self):
        pass

    def _get_csv_path(self, cfg):
        return os.path.join(
            cfg.get('dir_path', ''), "..", "datasets",
            cfg.get('dataset_name', ''), "modified_dataset", cfg.get('file_name', '')
        )

    @staticmethod
    def _hex_id_to_11_bits(hex_str):
        try:
            val = int(hex_str, 16)
            return [(val >> i) & 1 for i in range(10, -1, -1)]
        except (ValueError, TypeError):
            return [0] * 11

    def _load_csv(self, csv_path):
        data_bits, labels = [], []
        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2 or parts[0].lower() == "timestamp":
                    continue
                flag = parts[-1].strip().upper()
                if flag not in ("R", "T"):
                    continue
                data_bits.append(self._hex_id_to_11_bits(parts[1]))
                labels.append(1 if flag == "T" else 0)
        return data_bits, labels

    def _load_windows(self, csv_path):
        data_bits, labels = self._load_csv(csv_path)
        X = np.array(data_bits, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)
        n_wins = len(X) // self.WINDOW_SIZE
        limit = n_wins * self.WINDOW_SIZE
        X = X[:limit].reshape(n_wins, self.WINDOW_SIZE, 11)
        y_seq = y[:limit].reshape(n_wins, self.WINDOW_SIZE)
        y_win = y_seq.max(axis=1)
        return X, y_win
