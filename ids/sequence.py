import os
import pickle
import re

import numpy as np

from ids.base import IDS



_CANDUMP_ID_RE = re.compile(r'\bID:\s*([0-9a-fA-F]+)')

_ATTACK_TOKENS = {"T", "A", "ATTACK", "1"}
_BENIGN_TOKENS = {"R", "B", "NORMAL", "0"}


def _normalise_id(raw):
    
    return str(raw).strip().zfill(4).lower()


class Seq(IDS):

    def __init__(self):
        self.transition_matrix = {}   # can_id -> [bool, ...] over unique_ids
        self.unique_ids = []          # column order
        self._id_index = {}           # can_id -> column, rebuilt on load
        self._matrix = None           # unique_ids-ordered bool ndarray

    def _project_root(self, cfg):
        return os.path.normpath(os.path.join(cfg.get('dir_path', '.'), ".."))

    def _dataset_dir(self, cfg):
        return os.path.join(self._project_root(cfg), "datasets",
                            cfg.get('dataset_name', ''))

    def _trace_path(self, cfg):
        
        file_name = cfg.get('file_name', '')
        stem = file_name[:-4] if file_name.endswith(".csv") else file_name
        path = os.path.join(self._dataset_dir(cfg), "modified_dataset",
                            stem + ".csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Seq] no such trace: {path}")
        return os.path.normpath(path)

    @staticmethod
    def _looks_like_candump(path):
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    return "ID:" in line
        return False

    @classmethod
    def _read_candump(cls, path):
        """-> (ids, labels). candump logs here are attack-free, so labels are 0.

        Order is the whole point: the table is built from adjacent pairs, so
        no sorting or deduplication may happen on the way in.
        """
        ids = []
        with open(path, "r") as f:
            for line in f:
                match = _CANDUMP_ID_RE.search(line)
                if match:
                    ids.append(_normalise_id(match.group(1)))
        return ids, np.zeros(len(ids), dtype=int)

    @classmethod
    def _read_csv(cls, path):
        
        ids, labels = [], []
        unmapped = 0
        with open(path, "r") as f:
            for lineno, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                can_id = parts[1].strip()
                if not can_id:
                    continue                       # blank/truncated row
                if lineno == 0:
                    try:
                        float(parts[0])
                    except ValueError:
                        continue                   # header row
                ids.append(_normalise_id(can_id))
                token = parts[-1].strip().upper()
                if token in _ATTACK_TOKENS:
                    labels.append(1)
                elif token in _BENIGN_TOKENS:
                    labels.append(0)
                else:
                   
                    unmapped += 1
                    labels.append(0)
        if unmapped:
            print(f"  [Seq] WARNING: {unmapped} row(s) had an unrecognised "
                  f"label field and were counted as benign -- does "
                  f"{os.path.basename(path)} have a label column?")
        return ids, np.asarray(labels, dtype=int)

    @classmethod
    def _read_trace(cls, path):
        """-> (ids, labels), format sniffed from the first non-empty line."""
        if cls._looks_like_candump(path):
            return cls._read_candump(path)
        return cls._read_csv(path)

    def _rebuild_index(self):
        
        self._id_index = {cid: i for i, cid in enumerate(self.unique_ids)}
        if self.unique_ids:
            self._matrix = np.array(
                [self.transition_matrix[cid] for cid in self.unique_ids],
                dtype=bool)
        else:
            self._matrix = np.zeros((0, 0), dtype=bool)

    def score_ids(self, ids):
        
        if self._matrix is None:
            self._rebuild_index()
        if not len(ids):
            return np.empty(0, dtype=int)

        idx = np.fromiter((self._id_index.get(c, -1) for c in ids),
                          dtype=np.int64, count=len(ids))
        known = idx >= 0

        preds = np.ones(len(ids), dtype=int)
        preds[0] = 0 if known[0] else 1
        if len(ids) > 1:
            prev, curr = idx[:-1], idx[1:]
            pair_known = known[:-1] & known[1:]
            allowed = np.zeros(len(prev), dtype=bool)
            allowed[pair_known] = self._matrix[prev[pair_known],
                                               curr[pair_known]]
            preds[1:] = (~allowed).astype(int)
        return preds

    def train(self, train_dataset_dir=None, X_train=None, Y_train=None,
              cfg=None, **kwargs):
        
        cfg = cfg or {}
        path = self._trace_path(cfg)
        ids, labels = self._read_trace(path)

        print(f"Loading benign CAN log: {path}")
        if not ids:
            raise ValueError(
                f"[Seq] no CAN IDs parsed from {path} -- unexpected format?")

        if labels.any():
            keep = labels == 0
            print(f"  [Seq] dropping {int((~keep).sum())} tampered row(s); "
                  f"transitions are learned from the {int(keep.sum())} benign "
                  f"row(s) only")
            ids = [cid for cid, k in zip(ids, keep) if k]

        self.unique_ids = sorted(set(ids))
        self._id_index = {cid: i for i, cid in enumerate(self.unique_ids)}
        n = len(self.unique_ids)
        matrix = np.zeros((n, n), dtype=bool)

        src = np.fromiter((self._id_index[c] for c in ids[:-1]),
                          dtype=np.int64, count=len(ids) - 1)
        dst = np.fromiter((self._id_index[c] for c in ids[1:]),
                          dtype=np.int64, count=len(ids) - 1)
        matrix[src, dst] = True

        self._matrix = matrix
        self.transition_matrix = {cid: matrix[i].tolist()
                                  for i, cid in enumerate(self.unique_ids)}

        n_pairs = int(matrix.sum())
        print(f"Packets parsed    : {len(ids)}")
        print(f"Unique CAN IDs    : {n}")
        print(f"Transition pairs  : {n_pairs} of {n * n} possible "
              f"({n_pairs / max(n * n, 1):.1%} of the ID graph)")

    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs):
        
        cfg = cfg or {}
        if not self.unique_ids:
            raise RuntimeError(
                "[Seq] no transition table loaded -- run Stage 4 (training) "
                "or point testing.model_name at a saved model first.")

        path = self._trace_path(cfg)
        print(f"\nLoading test traffic: {path}")
        ids, labels = self._read_trace(path)
        if not ids:
            raise ValueError(f"[Seq] no CAN IDs parsed from {path}")

        preds = self.score_ids(ids)

        unseen = sorted({c for c in set(ids) if c not in self._id_index})
        n_unseen_pkts = sum(1 for c in ids if c not in self._id_index)
        print(f"  {len(ids)} packets, {len(set(ids))} distinct IDs "
              f"({len(self.unique_ids)} in the trained table)")
        if unseen:
            print(f"  {len(unseen)} ID(s) never seen benign -> auto-flagged "
                  f"({n_unseen_pkts} packets): {', '.join(unseen[:10])}"
                  f"{' ...' if len(unseen) > 10 else ''}")
        print(f"  flagged {int(preds.sum())} / {len(preds)} packets; "
              f"labelled attack {int((labels == 1).sum())}")

        return preds, labels

    def predict(self, X_test=None, **kwargs):
      
        if X_test is None:
            raise ValueError("[Seq] predict() needs an input")

        arr = np.asarray(X_test)

        if arr.ndim <= 1:
            if arr.dtype.kind in "iuf":        # numeric arbitration IDs
                return self.score_ids(
                    [_normalise_id(format(int(v), "x")) for v in arr])
            return self.score_ids([_normalise_id(v) for v in np.ravel(arr)])

        if arr.ndim == 4:
            arr = arr[..., 0]
        frames = arr[None, ...] if arr.ndim == 2 else arr

        preds = []
        for frame in frames:
            ids = [_normalise_id(
                       format(int("".join(str(int(b)) for b in row), 2), "x"))
                   for row in frame]
            preds.append(int(self.score_ids(ids).max()))
        return np.asarray(preds, dtype=int)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"transition_matrix": self.transition_matrix,
                         "unique_ids": self.unique_ids}, f)
        print(f"  Seq transition table saved to {path} "
              f"({len(self.unique_ids)} IDs, "
              f"{sum(sum(r) for r in self.transition_matrix.values())} pairs)")

    def load(self, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.transition_matrix = state["transition_matrix"]
        self.unique_ids = list(state["unique_ids"])
        self._rebuild_index()
        print(f"  Seq transition table loaded from {path} "
              f"({len(self.unique_ids)} IDs, {int(self._matrix.sum())} pairs)")
