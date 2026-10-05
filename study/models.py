"""LSTM forecaster, LSTM autoencoder and a shared training loop (paper Sec. III-F/G)."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

from . import config as C

torch.set_num_threads(4)


def seed_everything(seed=C.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMForecaster(nn.Module):
    """Two stacked LSTM layers, dropout after each, dense head emitting P(t+1..t+h)."""

    def __init__(self, n_in, hidden=C.FC_HIDDEN, dropout=C.FC_DROPOUT, horizon=C.FC_HORIZON):
        super().__init__()
        self.l1 = nn.LSTM(n_in, hidden[0], batch_first=True)
        self.l2 = nn.LSTM(hidden[0], hidden[1], batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden[1], horizon)

    def forward(self, x):
        h, _ = self.l1(x)
        h, _ = self.l2(self.drop(h))
        return self.head(self.drop(h[:, -1]))


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder.

    encoder LSTM -> LSTM bottleneck (latent vector) -> repeat over time ->
    decoder LSTM -> LSTM -> time-distributed dense (reconstructed features)
    """

    def __init__(self, n_feat, window=C.AE_WINDOW, hidden=C.AE_HIDDEN, latent=C.AE_LATENT,
                 dropout=C.AE_DROPOUT):
        super().__init__()
        self.window = window
        self.enc1 = nn.LSTM(n_feat, hidden, batch_first=True)
        self.enc2 = nn.LSTM(hidden, latent, batch_first=True)
        self.dec1 = nn.LSTM(latent, latent, batch_first=True)
        self.dec2 = nn.LSTM(latent, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_feat)  # applied per time step = TimeDistributed(Dense)

    def forward(self, x):
        h, _ = self.enc1(x)
        _, (z, _) = self.enc2(self.drop(h))
        z = z[-1].unsqueeze(1).repeat(1, self.window, 1)
        h, _ = self.dec1(z)
        h, _ = self.dec2(h)
        return self.out(self.drop(h))


def masked_mse(pred, target, mask):
    se = (pred - target) ** 2 * mask
    return se.sum() / mask.sum().clamp(min=1)


def train(model, batches_fn, n_train, val_fn, epochs, lr, batch_size, loss_fn, log_name):
    """Generic loop. batches_fn(idx) -> tensors; keeps the best-validation weights."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(C.SEED)
    best, best_state, history = np.inf, None, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(n_train)
        tot, n = 0.0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch = batches_fn(idx)
            opt.zero_grad()
            loss = loss_fn(model, *batch)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        model.eval()
        with torch.no_grad():
            val = val_fn(model)
        history.append({"epoch": ep, "train_loss": tot / n, "val_loss": val, "sec": time.time() - t0})
        flag = ""
        if val < best:
            best, best_state, flag = val, copy.deepcopy(model.state_dict()), " *"
        print(f"[{log_name}] epoch {ep:2d}  train {tot / n:.4f}  val {val:.4f}  "
              f"({time.time() - t0:.0f}s){flag}", flush=True)
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict(model, batches_fn, n, batch_size=4096):
    model.eval()
    outs = []
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n))
        outs.append(model(batches_fn(idx)[0]).numpy())
    return np.concatenate(outs) if outs else np.zeros((0,))
