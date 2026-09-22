import copy
import math
import os
import pickle
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ids.base import IDS



class SIGN_TYPE(Enum):
    PHYSVAL = "PHYSVAL"
    COUNTER = "COUNTER"
    CRC = "CRC"
    BINARY = "BINARY"


def _read_bitflip_stats(payloads, dlc):
    bit_flip = [0 for _ in range(dlc * 8)]
    magnitude = [0 for _ in range(dlc * 8)]
    payload_len = len(payloads)
    skipped_positions = 0

    for i in range(1, len(payloads)):
        payload = payloads[i]
        previous = payloads[i - 1]
        n = min(len(payload), len(previous), dlc * 8)
        for j in range(n):
            if payload[j] != previous[j]:
                bit_flip[j] = bit_flip[j] + 1
        if n < dlc * 8:
            skipped_positions += (dlc * 8 - n)

    if skipped_positions:
        print(f"[Candito/READ] {skipped_positions} bit-position comparisons "
              f"skipped (non-canonical-DLC frame for this ID) out of "
              f"{(payload_len - 1) * dlc * 8} total")

    for i in range(8 * dlc):
        bit_flip[i] = bit_flip[i] / payload_len
        magnitude[i] = math.ceil(math.log10(bit_flip[i])) if bit_flip[i] != 0 else float('-inf')

    return bit_flip, magnitude


def _read_phase1(magnitude, dlc):
    ref = []
    prev_magnitude = magnitude[0]
    ix_s = 0

    for ix in range(dlc * 8):
        if magnitude[ix] < prev_magnitude:
            ref.append((ix_s, ix))
            ix_s = ix
        if math.isinf(magnitude[ix]):   # skip a bit that never flips
            ix_s += 1
        prev_magnitude = magnitude[ix]

    if ix_s != dlc * 8:
        ref.append((ix_s, dlc * 8))
    return ref


def _read_match_counter(bit_flip):
    candidate = 0
    for i in range(1, len(bit_flip)):
        if math.isclose(bit_flip[i], 2 * bit_flip[i - 1], rel_tol=0.01):
            if math.isclose(bit_flip[i], 1.0, rel_tol=0.001):
                return candidate, i + 1
        else:
            candidate = i
    return -1, -1


def _read_phase2(ref, bit_flip, magnitude):
    r_ref = []
    for sign in ref:
        ix_start, ix_end = sign
        if ix_start == ix_end:
            continue
        if ix_start + 1 == ix_end:
            r_ref.append([ix_start, ix_end, SIGN_TYPE.BINARY])
            continue

        mgt = magnitude[ix_start:ix_end]

        start_ctr, end_ctr = _read_match_counter(bit_flip[ix_start:ix_end])
        if start_ctr >= 0 and end_ctr >= 0:
            if start_ctr == 0 and end_ctr == (ix_end - ix_start):
                r_ref.append([ix_start, ix_end, SIGN_TYPE.COUNTER])
                continue
            else:
                r_ref.append([ix_start + start_ctr, ix_start + end_ctr, SIGN_TYPE.COUNTER])
                ref.append((ix_start, ix_start + start_ctr))
                ref.append((ix_start + end_ctr, ix_end))
                continue

        found_crc = False
        for start_crc in range(ix_start, ix_end):
            mu = np.mean(bit_flip[start_crc:ix_end])
            std = np.std(bit_flip[start_crc:ix_end])
            if sum(mgt[start_crc:ix_end]) == 0:
                if 0.5 - std <= mu <= 0.5 + std:
                    r_ref.append([start_crc, ix_end, SIGN_TYPE.CRC])
                    ref.append((ix_start, start_crc))
                    found_crc = True
                    break
        if not found_crc:
            r_ref.append([ix_start, ix_end, SIGN_TYPE.PHYSVAL])
    return r_ref


def _read(trace, verbose=True):
    """READ: discover per-CAN-ID signal boundaries from attack-free traffic.

    Returns dict: CAN id -> [[start, end, SIGN_TYPE], ...]
    """
    ids = list(set(trace['Id'].tolist()))

    if verbose:
        print(f"[Candito/READ] {trace.shape[0]} rows, {len(ids)} CAN IDs")

    subtraces = [trace[trace['Id'] == _id] for _id in ids]

    results = {}
    for i in (tqdm(range(len(subtraces))) if verbose else range(len(subtraces))):
        subtrace = subtraces[i]
        payloads = subtrace['Payload'].tolist()
        # Canonical DLC = the MODE across this ID's own rows (not simply
        # whichever row happens to appear first) -- see docstring above.
        dlc_counts = subtrace['Dlc'].value_counts()
        dlc = int(dlc_counts.idxmax())
        if len(dlc_counts) > 1:
            print(f"[Candito/READ] ID {ids[i]} has {len(dlc_counts)} distinct "
                  f"DLC values {dict(dlc_counts)}; using the modal DLC={dlc}")

        bit_flip, magnitude = _read_bitflip_stats(payloads, dlc)
        ref = _read_phase1(magnitude, dlc)
        r_ref = _read_phase2(ref, bit_flip, magnitude)
        results[ids[i]] = r_ref

    return results


_KEEP_TYPES = {SIGN_TYPE.PHYSVAL.value, SIGN_TYPE.BINARY.value}


@dataclass
class Signal:
    start: int          # start bit (inclusive)
    end: int             # end bit (exclusive)
    dtype: str           # SIGN_TYPE value
    max_val: int         # 2**(end-start) - 1  (bit-range max)
    obs_min: int = 0     # min value observed over benign traffic
    obs_max: int = 0     # max value observed over benign traffic

    @property
    def width(self):
        return self.end - self.start


def _fit_read(benign_trace):
    """Run READ on attack-free traffic, keep only PHYSVAL/BINARY signals.

    Returns dict: CAN id -> list[Signal] (only IDs with >=1 kept signal).
    """
    results = _read(benign_trace, verbose=True)
    groups = {i: sub for i, sub in benign_trace.groupby("Id", sort=False)}
    id_signals = {}
    for _id, signals in results.items():
        kept = []
        payloads = (groups[_id]["Payload"].to_numpy()
                    if _id in groups else np.empty(0))
        for start, end, stype in signals:
            if stype.value in _KEEP_TYPES:
                width = end - start
                sig = Signal(start, end, stype.value, (1 << width) - 1)
                slices = [p[start:end] for p in payloads
                          if len(p) >= end]
                if slices:
                    vals = np.fromiter((int(b, 2) for b in slices),
                                       dtype=np.int64, count=len(slices))
                    sig.obs_min = int(vals.min())
                    sig.obs_max = int(vals.max())
                kept.append(sig)
        if kept:   # ignore IDs whose payload is entirely constant/counter/crc
            id_signals[_id] = kept
    print(f"[Candito] READ kept feature signals for {len(id_signals)} / "
          f"{len(results)} CAN IDs")
    return id_signals


def _encode_id(sub_trace, signals, scaling="bitrange"):
    """Encode one CAN ID's packets into a [n_packets, n_signals] float matrix,
    each signal rescaled to [0, 1] (paper Sec. 4.1).

    scaling:
      "bitrange" (default) -- value / (2**width - 1). Deterministic, no fit.
      "minmax"             -- (value - obs_min) / (obs_max - obs_min), fit on
                              benign traffic and clamped to [0, 1].
    """
    payloads = sub_trace["Payload"].to_numpy()
    n = len(payloads)
    k = len(signals)
    X = np.zeros((n, k), dtype=np.float32)
    short_rows = 0
    for j, sig in enumerate(signals):
        def _bits(p, sig=sig):
            nonlocal short_rows
            b = p[sig.start:sig.end]
            if len(b) < sig.width:
                short_rows += 1
                if not b:
                    return 0
            return int(b, 2)
        col = np.fromiter((_bits(p) for p in payloads), dtype=np.float64, count=n)
        if scaling == "minmax":
            rng = sig.obs_max - sig.obs_min
            X[:, j] = (np.clip((col - sig.obs_min) / rng, 0.0, 1.0)
                       if rng > 0 else 0.0)
        else:  # bitrange
            X[:, j] = (col / sig.max_val) if sig.max_val > 0 else 0.0
    if short_rows:
        print(f"[Candito] WARNING: {short_rows} payload/signal slices were "
              f"shorter than expected (mixed-DLC CAN ID) and treated as 0")
    return X


def _encode_trace(trace, id_signals, scaling="bitrange"):
    """Encode a full trace, grouped by CAN ID.

    Returns dict: id -> {"X": [n,k] float32, "y": [n] int (IsTampered)}.
    Only IDs present in `id_signals` (i.e. with feature signals) are kept.
    """
    out = {}
    has_label = "IsTampered" in trace.columns
    for _id, sub in trace.groupby("Id", sort=False):
        if _id not in id_signals:
            continue
        sub = sub.reset_index(drop=True)
        X = _encode_id(sub, id_signals[_id], scaling)
        y = (sub["IsTampered"].to_numpy().astype(int) if has_label
             else np.zeros(len(sub), dtype=int))
        out[_id] = {"X": X, "y": y}
    return out


WINDOW = 40   # window length n = 40 packets (paper)


def _make_windows(X, y, window=WINDOW, stride=1):
    """Return (Xw, yw): [num, window, k] float32 and [num] int window labels."""
    n = len(X)
    if n < window:
        return np.empty((0, window, X.shape[1]), np.float32), np.empty((0,), int)
    idx = range(0, n - window + 1, stride)
    Xw = np.stack([X[i:i + window] for i in idx]).astype(np.float32)
    yw = np.array([int(y[i:i + window].any()) for i in idx], dtype=int)
    return Xw, yw


def _train_windows(X, y, window=WINDOW):
    """Benign-only, stride-1 windows for training/threshold estimation."""
    Xw, yw = _make_windows(X, y, window, stride=1)
    return Xw[yw == 0]


def _test_windows(X, y, window=WINDOW):
    """Non-overlapping windows for evaluation (stride = window)."""
    return _make_windows(X, y, window, stride=window)



class LSTMAutoencoder(nn.Module):
    """
    Encoder:  Dense(k->128, ELU) -> Dropout(0.2)
              -> LSTM(128->64) -> LSTM(64->64)          [returns last (h, c)]
    Sequence reversal of the encoded output (Sutskever trick, mitigates
    underfit). Decoder: LSTM(64->64, init with encoder (h,c)) -> LSTM(64->64)
              -> Dense(64->128, ELU) -> Dense(128->k, Sigmoid)
    """

    def __init__(self, input_dim, hidden_dim=64, dense_dim=128, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim

        self.enc_dense = nn.Linear(input_dim, dense_dim)
        self.enc_elu = nn.ELU()
        self.enc_dropout = nn.Dropout(dropout)
        self.enc_lstm1 = nn.LSTM(dense_dim, hidden_dim, batch_first=True)
        self.enc_lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.dec_lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dec_lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dec_dense1 = nn.Linear(hidden_dim, dense_dim)
        self.dec_elu = nn.ELU()
        self.dec_dense2 = nn.Linear(dense_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                          # x: [B, T, k]
        h = self.enc_dropout(self.enc_elu(self.enc_dense(x)))
        h, _ = self.enc_lstm1(h)
        enc, (h_n, c_n) = self.enc_lstm2(h)        # enc: [B, T, hidden]

        enc_rev = torch.flip(enc, dims=[1])        # reverse time axis

        d, _ = self.dec_lstm1(enc_rev, (h_n, c_n))
        d, _ = self.dec_lstm2(d)
        d = self.dec_elu(self.dec_dense1(d))
        return self.sigmoid(self.dec_dense2(d))    # [B, T, k] in [0,1]



def _auto_eval_batch_size(X, device, fallback):
    is_gpu = getattr(device, "type", str(device)) in ("cuda", "mps")
    if not is_gpu or X.ndim < 3:
        return fallback
    per_window_elems = max(int(X.shape[1]) * max(int(X.shape[2]), 1), 1)
    target_elems = 8_000_000
    return int(max(fallback, min(8192, target_elems // per_window_elems)))


@torch.no_grad()
def _mean_mse_batched(model, X, device, batch_size):
    """Mean MSE of the autoencoder over X, computed in mini-batches (avoids
    holding the whole validation split on the GPU at once)."""
    model.eval()
    total_sq = torch.zeros((), device=device)
    total_elems = 0
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        recon = model(xb)
        total_sq += ((recon - xb) ** 2).sum()
        total_elems += xb.numel()
    return (total_sq / total_elems).item() if total_elems else 0.0


def _train_id_model(Xtrain, input_dim, device, epochs=50, patience=5,
                     batch_size=64, lr=1e-3, val_frac=0.1, seed=0, Xval=None):
    """Train one autoencoder on benign windows Xtrain: [N, T, k].

    Early stopping is monitored on a held-out benign split: the caller's
    `Xval` if given (a dedicated benign validation source), else a
    `val_frac` tail carved off Xtrain itself.
    """
    torch.manual_seed(seed)
    n = len(Xtrain)
    if n == 0:
        return None, {"train": [], "val": []}

    if Xval is not None and len(Xval):
        Xtr = Xtrain
    else:
        n_val = max(1, int(n * val_frac)) if n > 10 else 0
        Xtr = Xtrain[:n - n_val] if n_val else Xtrain
        Xval = Xtrain[n - n_val:] if n_val else None

    is_cuda = getattr(device, "type", str(device)) == "cuda"
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(Xtr, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, pin_memory=is_cuda,
    )
    eval_batch_size = (_auto_eval_batch_size(Xval, device, batch_size)
                       if Xval is not None else batch_size)

    model = LSTMAutoencoder(input_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0

    for epoch in range(epochs):
        model.train()
        running = torch.zeros((), device=device)
        n_seen = 0
        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=is_cuda)
            opt.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            opt.step()
            running += loss.detach() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = (running / n_seen).item()

        if Xval is not None and len(Xval):
            val_loss = _mean_mse_batched(model, Xval, device, eval_batch_size)
            improved = val_loss < best_val - 1e-6
        else:
            val_loss = train_loss
            improved = train_loss < best_val - 1e-6

        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    model.load_state_dict(best_state)
    return model, None



THRESHOLD_PERCENTILE = 99.99

_GPU_EVAL_ELEMS = 8_000_000
_CPU_EVAL_BATCH = 256


def _score_batch_size(Xw, device):
    is_gpu = getattr(device, "type", str(device)) in ("cuda", "mps")
    if not is_gpu or Xw.ndim < 3:
        return _CPU_EVAL_BATCH
    per_window_elems = max(int(Xw.shape[1]) * max(int(Xw.shape[2]), 1), 1)
    return int(max(64, min(8192, _GPU_EVAL_ELEMS // per_window_elems)))


@torch.no_grad()
def _window_scores(model, Xw, device, batch_size=None):
    """Squared L2 reconstruction error per window."""
    if batch_size is None:
        batch_size = _score_batch_size(Xw, device)
    model.eval()
    scores = []
    for i in range(0, len(Xw), batch_size):
        xb = torch.as_tensor(Xw[i:i + batch_size], dtype=torch.float32, device=device)
        recon = model(xb)
        err = (recon - xb) ** 2
        scores.append(err.sum(dim=[1, 2]).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty((0,), np.float32)


def _compute_threshold(train_scores, percentile=THRESHOLD_PERCENTILE):
    return float(np.percentile(train_scores, percentile))


def _score_predict(test_scores, threshold):
    return (test_scores > threshold).astype(int)


class Candito(IDS):
    """CANdito: one LSTM-autoencoder + threshold per CAN ID."""

    EPOCHS = 50
    PATIENCE = 5
    BATCH_SIZE = 64
    LR = 1e-3
    THRESHOLD_PERCENTILE = THRESHOLD_PERCENTILE   # 99.99, paper Sec. 4.3
    MIN_TRAIN_WINDOWS = 50
    SCALING = "bitrange"

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.id_models = {}     # id -> {"state_dict", "input_dim", "threshold"}
        self.id_signals = {}    # id -> [Signal, ...], fit on benign training data only
        self.known_ids = set()  # every CAN ID seen anywhere in benign training
        self.window = WINDOW
        self.scaling = self.SCALING

    
    def _params(self, cfg):
        """Resolve the paper's hyper-parameters from the `training.Candito`
        block of config.yaml, falling back to the paper's own values.

        Mirrors the knobs of CANdito's own reference configs, so a run here
        can reproduce the paper's protocol exactly:
        window/patience/batch_size/lr/threshold_percentile/min_train_windows/
        scaling, plus the two dedicated benign splits (see train()).
        """
        block = (cfg.get('training', {}) or {}).get('Candito', {}) or {}
        return {
            'window': int(block.get('window', WINDOW)),
            'patience': int(block.get('patience', self.PATIENCE)),
            'batch_size': int(block.get('batch_size', self.BATCH_SIZE)),
            'lr': float(block.get('lr', self.LR)),
            'threshold_percentile': float(
                block.get('threshold_percentile', self.THRESHOLD_PERCENTILE)),
            'min_train_windows': int(
                block.get('min_train_windows', self.MIN_TRAIN_WINDOWS)),
            'scaling': block.get('scaling', self.SCALING),
            'threshold_file': block.get('threshold_file', None),
            'validation_file': block.get('validation_file', None),
        }

    
    def train(self, train_dataset_dir=None, X_train=None, Y_train=None, cfg=None, **kwargs):
        cfg = cfg or {}
        p = self._params(cfg)
        epochs = cfg.get('epochs', self.EPOCHS)
        df = self._load_cached_df(cfg)
        self.id_models = {}
        self.window = p['window']
        self.scaling = p['scaling']

        benign = df[df["IsTampered"] == 0].reset_index(drop=True)
        print(f"Candito — fitting READ on {len(benign)} benign rows "
              f"({df['Id'].nunique()} CAN IDs total)")
        self.id_signals = _fit_read(benign)
        self.known_ids = set(benign["Id"].unique())

        encoded = _encode_trace(df, self.id_signals, self.scaling)

        thr_enc = self._encode_split(cfg, p['threshold_file'], "threshold")
        val_enc = self._encode_split(cfg, p['validation_file'], "early-stopping validation")

        trained = 0
        dedicated_val_ids = fallback_val_ids = 0
        for _id, d in encoded.items():
            Xw = _train_windows(d["X"], d["y"], self.window)
            if len(Xw) < p['min_train_windows']:
                continue
            input_dim = d["X"].shape[1]

            if thr_enc is not None and _id in thr_enc:
                Xtrain = Xw
                thr_windows = _train_windows(thr_enc[_id]["X"], thr_enc[_id]["y"],
                                             self.window)
            else:
                cut = int(len(Xw) * 0.9)
                Xtrain, thr_windows = Xw[:cut], Xw[cut:]
            if len(thr_windows) == 0:
                thr_windows = Xtrain

            Xval = None
            if val_enc is not None and _id in val_enc:
                Xval_w = _train_windows(val_enc[_id]["X"], val_enc[_id]["y"],
                                        self.window)
                if len(Xval_w):
                    Xval = Xval_w
            if Xval is not None:
                dedicated_val_ids += 1
            else:
                fallback_val_ids += 1

            model, _ = _train_id_model(
                Xtrain, input_dim, self.device,
                epochs=epochs, patience=p['patience'],
                batch_size=p['batch_size'], lr=p['lr'], Xval=Xval,
            )
            if model is None:
                continue

            thr_scores = _window_scores(model, thr_windows, self.device)
            threshold = _compute_threshold(thr_scores, p['threshold_percentile'])

            self.id_models[_id] = {
                "state_dict": {k: v.detach().cpu()
                               for k, v in model.state_dict().items()},
                "input_dim": input_dim,
                "threshold": threshold,
            }
            trained += 1
            print(f"  id={_id}  windows={len(Xw)}  threshold={threshold:.4f}")

        print(f"Candito — trained {trained}/{len(encoded)} CAN IDs with a kept "
              f"signal ({len(encoded) - trained} had < {p['min_train_windows']} "
              f"benign windows); {dedicated_val_ids} used the dedicated "
              f"early-stopping split, {fallback_val_ids} fell back to a "
              f"tail-of-training split")

    
    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs):
        cfg = cfg or {}
        df = self._load_cached_df(cfg)
        encoded = _encode_trace(df, self.id_signals, self.scaling)

        all_preds, all_labels = [], []

        for _id, entry in self.id_models.items():
            if _id not in encoded:
                continue
            model = LSTMAutoencoder(entry["input_dim"]).to(self.device)
            model.load_state_dict(entry["state_dict"])

            Xte, yte = _test_windows(encoded[_id]["X"], encoded[_id]["y"], self.window)
            if len(Xte) == 0:
                continue
            scores = _window_scores(model, Xte, self.device)
            preds = _score_predict(scores, entry["threshold"])
            all_preds.append(preds)
            all_labels.append(yte)
            print(f"  [test] id={_id}  windows={len(Xte)}  attacks={int(yte.sum())}  "
                  f"flagged={int(preds.sum())}")

        unknown_ids = set(df["Id"].unique()) - self.known_ids
        for _id in unknown_ids:
            sub = df[df["Id"] == _id].reset_index(drop=True)
            y = sub["IsTampered"].to_numpy().astype(int)
            n = len(y)
            if n < self.window:
                continue
            idx = range(0, n - self.window + 1, self.window)
            yw = np.array([int(y[i:i + self.window].any()) for i in idx], dtype=int)
            preds = np.ones_like(yw)
            all_preds.append(preds)
            all_labels.append(yw)
            print(f"  [test] id={_id} NOT IN BENIGN TRAINING -> flagged as "
                  f"attack (windows={len(yw)})")

        if not all_preds:
            return np.array([], dtype=int), np.array([], dtype=int)
        return np.concatenate(all_preds), np.concatenate(all_labels)

    def predict(self, X_test=None, **kwargs):
        raise NotImplementedError(
            "Candito scores per CAN ID with its own autoencoder + threshold "
            "over windows of that ID's own traffic -- there is no single "
            "flat-feature prediction path. Use test(cfg=...) instead."
        )

    def save(self, path):
        state = {
            "id_models": {
                _id: {"state_dict": {k: v.detach().cpu()
                                     for k, v in e["state_dict"].items()},
                      "input_dim": e["input_dim"],
                      "threshold": e["threshold"]}
                for _id, e in self.id_models.items()
            },
            "id_signals": self.id_signals,
            "known_ids": self.known_ids,
            "window": self.window,
            "scaling": self.scaling,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Candito model saved to {path} "
              f"({len(self.id_models)} CAN-ID autoencoders)")

    def load(self, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.id_models = state["id_models"]
        self.id_signals = state["id_signals"]
        self.known_ids = state["known_ids"]
        self.window = state["window"]
        self.scaling = state.get("scaling", self.SCALING)
        print(f"  Candito model loaded from {path} "
              f"({len(self.id_models)} CAN-ID autoencoders)")

    
    def _encode_split(self, cfg, file_name, what):
        """Encode an optional dedicated benign split (threshold / validation)
        with the SAME fitted READ signals as training. Returns None when the
        split isn't configured, so the caller falls back to a split of the
        training windows."""
        if not file_name:
            print(f"Candito — no {what} file configured; falling back to a "
                  f"tail split of each ID's own training windows")
            return None
        df = self._load_cached_df(cfg, file_name)
        n_tampered = int(df["IsTampered"].sum())
        if n_tampered:
            print(f"Candito — WARNING: {what} file {file_name} contains "
                  f"{n_tampered} tampered rows; windows covering them are "
                  f"dropped (this split must be attack-free)")
        print(f"Candito — {what} split from {file_name} ({len(df)} rows)")
        return _encode_trace(df, self.id_signals, self.scaling)

    def _load_cached_df(self, cfg, file_name=None):
        file_name = file_name or cfg['file_name']
        prefix = file_name[:-4] if file_name.endswith(".csv") else file_name
        cache_path = os.path.join(
            cfg['dir_path'], "..", "datasets", cfg['dataset_name'],
            "features", "Candito", prefix + "_raw.pkl",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached Candito features at {cache_path} -- run Stage 1 "
                f"(dataset_processing.feature_extraction: true, "
                f"feature_extractor: CanditoExtractor) for file_name="
                f"{file_name} first."
            )
        return pd.read_pickle(cache_path)
