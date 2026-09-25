import os
import pandas as pd
import numpy as np
from collections import Counter
import math
from ids.base import IDS
from datetime import datetime
import pickle

class Shannon(IDS):

    def __init__(self, time_window: float = 0.032768, k_factor: float = 3.25):
        """
        Initializes the analysis attack.

        Args:
            time_window (float): The duration in seconds of the sliding window
                                 for entropy calculation.
            k_factor (float): The number of standard deviations from the mean
                              to set the anomaly threshold.
        """
        self.time_window = time_window
        self.k_factor = k_factor
        self.mean_h_ = None
        self.std_h_ = None
        super().__init__()

    # modified_dataset layout: timestamp,can_id,dlc,byte0..byte7,R/T flag
    CSV_COLUMNS = (["timestamp", "can_id", "dlc"]
                   + ["b%d" % i for i in range(1, 9)] + ["label"])

    def _is_hex(self, s):
        """Checks if a string can be interpreted as a hexadecimal."""
        try:
            int(str(s), 16)
            return True
        except (ValueError, TypeError):
            return False

    @classmethod
    def _read_can_csv(cls, path):
        """Read a modified_dataset CSV whether or not it carries a header row.

        Some files in modified_dataset/ have the `timestamp,can_id,dlc,...`
        header (DoS_test_3000.csv), others do not
        (car_hacking_normal_run_data.csv). Reading a headered file with
        header=None turned the header itself into a data row, so 'timestamp'
        ended up in the numeric timestamp column and the first window
        comparison died with "can only concatenate str (not float) to str".
        Sniff the first field instead and let pandas skip the header when
        there is one.
        """
        with open(path, "r") as f:
            first = f.readline().split(",")[0].strip().lower()
        has_header = first in ("timestamp", "time")
        df = pd.read_csv(path, header=0 if has_header else None,
                         names=cls.CSV_COLUMNS, low_memory=False)
        df = cls._repair_ragged_rows(df)
        # A header row is not the only way a stray non-numeric line gets in
        # (blank trailing lines, partially-written rows); drop whatever will
        # not coerce rather than carrying NaN timestamps into the windowing.
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        bad = int(df["timestamp"].isna().sum())
        if bad:
            print(f"  [Shannon] dropped {bad} row(s) with an unparseable timestamp")
            df = df[df["timestamp"].notna()]
        return df.reset_index(drop=True)

    @classmethod
    def _repair_ragged_rows(cls, df):
        """Recover the R/T flag on rows that are not padded to 8 byte fields.

        modified_dataset/ is not uniformly padded. DoS_dataset.csv left-pads a
        DLC=2 frame to eight fields (`...,00,00,00,00,00,00,01,00,R`), but
        DoS_test_3000.csv writes the two real bytes and stops
        (`...,05f0,2,01,00,R` -- 6 fields, 836 of them in that file). Read
        against a fixed 12-name schema, such a row puts the flag in b3 and
        leaves `label` NaN, which `_attack_flags` then reads as benign: a
        ragged *attack* row would be silently relabelled. So take the flag
        from the last populated field and blank the byte slots from there on.

        Byte ordering does not need repairing for this detector: the symbol
        is the byte value alone, so the left-padded and unpadded spellings of
        the same frame yield the same multiset and the same entropy.
        """
        byte_cols = ["b%d" % i for i in range(1, 9)]
        ragged = df.index[df["label"].isna()]
        if not len(ragged):
            return df
        for idx in ragged:
            populated = [v for v in df.loc[idx, byte_cols] if pd.notna(v)]
            if not populated:
                continue
            df.at[idx, "label"] = populated[-1]
            for offset in range(len(populated) - 1, len(byte_cols)):
                df.at[idx, byte_cols[offset]] = np.nan
        print(f"  [Shannon] recovered the R/T flag on {len(ragged)} unpadded "
              f"row(s) (fewer than 8 byte fields)")
        return df

    def _byte_values(self, df):
        """Add the per-frame list of 8 payload byte values (hex -> int)."""
        raw = df[["b%d" % i for i in range(1, 9)]].values.tolist()
        df["Byte_Values"] = [
            [int(b, 16) if self._is_hex(b) else 0 for b in row] for row in raw
        ]
        return df

    def _window_bounds(self, timestamps):
        """-> (starts, stops) index pairs for non-empty fixed-width time bins.

        Bins are laid from the first timestamp, the same scheme
        `_get_window_entropies` walks, but derived arithmetically instead of
        by re-scanning the whole frame per window. The old
        `df[(ts >= t) & (ts < t + w)]` mask was O(windows x frames) -- on
        DoS_test.csv (1.4M frames, ~36k windows) that is ~5e10 comparisons.
        This is O(frames log frames) and, unlike repeated `t += window`,
        does not accumulate float drift across tens of thousands of bins.
        """
        ts = np.asarray(timestamps, dtype=np.float64)
        order = np.argsort(ts, kind="stable")
        ts_sorted = ts[order]
        bins = np.floor((ts_sorted - ts_sorted[0]) / self.time_window).astype(np.int64)
        edges = np.flatnonzero(np.diff(bins)) + 1
        starts = np.concatenate(([0], edges))
        stops = np.concatenate((edges, [len(ts_sorted)]))
        return order, starts, stops

    def _window_entropies(self, df, benign_only=False):
        """-> (entropy per non-empty window, window attack label).

        A window is labelled attack (1) if it contains any tampered frame,
        matching how this detector actually decides: the score is a property
        of the whole window, so the window is the unit that can be right or
        wrong. With `benign_only`, attack windows are dropped instead.
        """
        order, starts, stops = self._window_bounds(df["timestamp"].to_numpy())
        byte_lists = df["Byte_Values"].to_numpy()[order]
        is_attack = self._attack_flags(df)[order]

        entropies, labels = [], []
        for start, stop in zip(starts, stops):
            all_bytes = [b for row in byte_lists[start:stop] for b in row]
            if not all_bytes:
                continue
            label = int(is_attack[start:stop].any())
            if benign_only and label:
                continue
            entropies.append(self._calculate_shannon_entropy(all_bytes))
            labels.append(label)
        return np.asarray(entropies, dtype=np.float64), np.asarray(labels, dtype=int)

    @staticmethod
    def _attack_flags(df):
        """Per-frame tampered flag from the label column (T/A/ATTACK/1)."""
        if "label" not in df.columns:
            return np.zeros(len(df), dtype=bool)
        return (df["label"].astype(str).str.strip().str.upper()
                .isin(("T", "A", "ATTACK", "1")).to_numpy())

    def window_scores(self, entropies):
        """Anomaly score: absolute deviation from the benign mean, in sigmas.

        Thresholding this at k reproduces the two-sided rule
        `not (lower <= H <= upper)` exactly, but is monotone in
        anomalousness, so it can be fed to an AUC or a threshold sweep. Raw
        entropy cannot: the benign class sits in the middle of its range,
        so neither direction alone orders the classes.
        """
        if self.std_h_ in (None, 0) or self.mean_h_ is None:
            raise RuntimeError("You must call a `fit` method before scoring.")
        return np.abs(entropies - self.mean_h_) / self.std_h_

    def _calculate_shannon_entropy(self, data_list: list) -> float:
        """Calculates the Shannon entropy for a list of byte values."""
        if not data_list:
            return 0.0
        counts = Counter(data_list)
        total_symbols = len(data_list)
        entropy = -sum((c / total_symbols) * math.log2(c / total_symbols) for c in counts.values())
        return entropy

    def fit_from_csv(self, normal_data_csv_path: str, benign_only: bool = True):
        """
        CUSTOM FIT METHOD: Learns the entropy baseline from a normal data CSV file.
        This must be called before 'apply'.

        With `benign_only` (the default), any window containing a tampered
        frame is excluded, so a *labelled* capture can serve as the
        calibration source. That matters here: the only attack-free file in
        modified_dataset/ (car_hacking_normal_run_data.csv, timestamps
        ~1479121434) is a different capture session from the attack traces
        (~1478200247, roughly ten days earlier), and window entropy shifts
        between sessions -- calibrating across that gap put this trace's own
        benign windows 3.15 sigma from the mean before any attack, i.e. a
        44% false-positive rate at k=3.25. Fitting on the benign rows of a
        same-session file (e.g. DoS_dataset.csv) removes that offset.
        """
        print(f"Fitting model from '{os.path.basename(normal_data_csv_path)}'...")

        df_normal = self._read_can_csv(normal_data_csv_path)
        df_normal = self._byte_values(df_normal)

        normal_entropies, _ = self._window_entropies(
            df_normal, benign_only=benign_only)

        if not len(normal_entropies):
            raise ValueError("Could not calculate entropy from the provided normal data.")

        self.mean_h_ = float(np.mean(normal_entropies))
        self.std_h_ = float(np.std(normal_entropies))
        print(f"Fit complete. Baseline Mean Entropy: {self.mean_h_:.4f}, "
              f"Std Dev: {self.std_h_:.4f} "
              f"({len(normal_entropies)} benign windows, "
              f"window={self.time_window}s)")
        print(f"  Accept band: [{self.mean_h_ - self.k_factor * self.std_h_:.4f}, "
              f"{self.mean_h_ + self.k_factor * self.std_h_:.4f}]  (k={self.k_factor})")

    def _params(self, cfg):
        """Resolve time_window / k_factor / benign_only from the
        `training.Shannon` block of config.yaml, mirroring how ids/candito.py
        reads `training.Candito`. Absent keys keep the constructor defaults,
        so an existing config that has no Shannon block behaves as before."""
        block = (cfg.get('training', {}) or {}).get('Shannon', {}) or {}
        if 'time_window' in block:
            self.time_window = float(block['time_window'])
        if 'k_factor' in block:
            self.k_factor = float(block['k_factor'])
        return {'benign_only': bool(block.get('benign_only', True)),
                'sweep_k': bool(block.get('sweep_k', True))}

    def _dataset_csv(self, cfg):
        return os.path.join(
            cfg.get('dir_path', ''), "..", "datasets", cfg.get('dataset_name', ''),
            "modified_dataset", cfg.get('file_name', '')[:-4] + ".csv"
        )

    def train(self, train_dataset_dir=None, X_train=None, Y_train=None, cfg=None, **kwargs):
        cfg = cfg or {}
        p = self._params(cfg)
        self.fit_from_csv(self._dataset_csv(cfg), benign_only=p['benign_only'])


    def prepare_frame_list(self, cfg=None):
        cfg = cfg or {}
        attack_data_path = os.path.join(
            cfg.get('dir_path', ''), "..", "datasets", cfg.get('dataset_name', ''),
            "modified_dataset", cfg.get('file_name', '')[:-4] + ".csv"
        )
        print(f"\nLoading attack data from '{os.path.basename(attack_data_path)}' for 'apply' method...")

        df_attack = self._byte_values(self._read_can_csv(attack_data_path))
        df_attack['data'] = df_attack['Byte_Values'].apply(bytes)

        attack_frames = df_attack[['timestamp', 'data', 'label']].to_dict('records')
        print(f"Converted {len(attack_frames)} rows into 'frames' format.")
        return attack_frames

    def apply(self, frames: list[dict], **kwargs) -> list[dict]:
        """
        Analyzes frames for anomalies and adds an 'anomaly_detected' key.
        This method conforms to the StatisticalAttack structure.
        """
        if self.mean_h_ is None or self.std_h_ is None:
            raise RuntimeError("You must call a `fit` method before using `apply`.")

        if not frames:
            return []

        print("Applying entropy analysis...")
        adv_frames = [f.copy() for f in frames]
        df_test = pd.DataFrame(adv_frames)
        df_test['Byte_Values'] = df_test['data'].apply(lambda x: list(x))

        lower_thresh = self.mean_h_ - self.k_factor * self.std_h_
        upper_thresh = self.mean_h_ + self.k_factor * self.std_h_

        start_time, end_time = df_test['timestamp'].min(), df_test['timestamp'].max()
        current_ts = start_time
        num_anomalies_found = 0

        while current_ts < end_time:
            window_end = current_ts + self.time_window
            window_indices = df_test.index[
                (df_test['timestamp'] >= current_ts) & (df_test['timestamp'] < window_end)
            ].tolist()

            if window_indices:
                window_df = df_test.loc[window_indices]
                all_bytes = [byte for byte_list in window_df['Byte_Values'] for byte in byte_list]

                is_anomaly = False
                if all_bytes:
                    entropy_val = self._calculate_shannon_entropy(all_bytes)
                    if not (lower_thresh <= entropy_val <= upper_thresh):
                        is_anomaly = True
                        num_anomalies_found += 1

                for idx in window_indices:
                    adv_frames[idx]['anomaly_detected'] = is_anomaly
            current_ts = window_end

        print(f"Analysis complete. Found {num_anomalies_found} anomalous windows.")
        return adv_frames

    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs):
        """Score a trace and return WINDOW-level (preds, labels).

        Predictions used to be returned per frame, by copying each window's
        verdict onto all ~39 frames inside it and comparing that against
        per-frame labels. That measures the wrong thing: a window holding one
        injected frame among 39 counts as 38 false positives even though the
        detector's decision -- "this window's entropy is off" -- was right,
        and it cannot be right at frame granularity because it never looks at
        a frame in isolation. On DoS_test_3000.csv the identical predictions
        scored F1 0.479 per frame and 0.872 per window.

        `self.window_scores_` / `self.window_labels_` are left on the
        instance afterwards, so a caller can compute AUC or re-threshold
        without rescoring (src/test.py only forwards preds and labels).
        """
        cfg = cfg or {}
        p = self._params(cfg)
        if self.mean_h_ is None or self.std_h_ is None:
            raise RuntimeError("You must call a `fit` method before using `test`.")

        path = self._dataset_csv(cfg)
        print(f"\nLoading attack data from '{os.path.basename(path)}' ...")
        df = self._byte_values(self._read_can_csv(path))

        entropies, labels = self._window_entropies(df, benign_only=False)
        scores = self.window_scores(entropies)
        preds = (scores > self.k_factor).astype(int)

        self.window_entropies_ = entropies
        self.window_scores_ = scores
        self.window_labels_ = labels

        print(f"  {len(df)} frames -> {len(labels)} windows "
              f"({len(df) / max(len(labels), 1):.1f} frames/window); "
              f"benign={int((labels == 0).sum())} attack={int((labels == 1).sum())}")
        if (labels == 1).any() and (labels == 0).any():
            print(f"  benign entropy median={np.median(entropies[labels == 0]):.4f}  "
                  f"attack entropy median={np.median(entropies[labels == 1]):.4f}")
            try:
                from sklearn.metrics import roc_auc_score
                print(f"  window-level AUC (|H-mean|/std) = "
                      f"{roc_auc_score(labels, scores):.4f}")
            except Exception as exc:      # sklearn absent / single class
                print(f"  AUC unavailable: {exc}")
            if p['sweep_k']:
                print("    k      recall      FPR")
                for k in (1, 2, 3, self.k_factor, 5, 8, 12, 20):
                    flagged = scores > k
                    print(f"    {k:<5g}  {flagged[labels == 1].mean():7.2%}  "
                          f"{flagged[labels == 0].mean():7.2%}")

        return preds, labels

    def predict(self, X_test=None, **kwargs):
        pass

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'mean_h_': self.mean_h_,
                'std_h_': self.std_h_,
                'time_window': self.time_window,
                'k_factor': self.k_factor
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.mean_h_ = data['mean_h_']
        self.std_h_ = data['std_h_']
        self.time_window = data['time_window']
        self.k_factor = data['k_factor']
