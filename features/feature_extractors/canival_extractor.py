"""
CanivalExtractor — feature extraction for the CANival IDS
=========================================================
Builds everything ids/canival.py consumes, for whichever dataset
`dataset_name` selects. One extractor rather than three because the pipeline
picks a single `feature_extractor` per run; the per-dataset logic is unchanged
and lives in _extract_syncan / _extract_xcanids / _extract_road.

Two feature families are produced, one per detector:

  TIL    Gaussian (mu, sigma) per CAN ID fitted on benign inter-arrival gaps,
         plus the log-likelihood range used to normalise scores into z_itv.
         Written as features/til/*_norm_dist.json + one z_itv parquet per
         eval file. TIL itself has no trainable weights, so this IS its fit.
  CANet  per-CAN-ID sliding windows of normalised signals. Materialised here
         for SynCAN (.npz) and ROAD (.parquet); for X-CANIDS the sig107
         parquets are large and are windowed at inference time instead, so
         only the TIL half runs here.

Outputs, under datasets/<dataset_name>/features/ :

  SynCAN    canet/syncan_canet_{stem}.npz
            til/syncan_til_norm_dist.json, til/syncan_til_{stem}.parquet
  X-CANIDS  til/xcanids_til_norm_dist.json, til/xcanids_til_{stem}.parquet
            (CANet reads modified_dataset/sig107_dump*.parquet directly)
  ROAD      road_config.json  (ID selection, ID_MPS/ID_NSIG, per-signal norms)
            canet/{train,valid,test}_{stem}.parquet
            til/road_til_norm_dist.json, til/road_til_{stem}.parquet

Existing outputs are skipped, so a re-run is cheap and interrupted runs resume.

Consolidated from syncan_canet.py, xcanids_extractor.py and road_extractor.py
of the CAN-Rakshak CANival build; logic is unchanged. Module-level helpers are
prefixed _syn_ / _xc_ / _road_ because the three originals used clashing names
for differently-behaving functions (get_time_intervals, build_norm_dist,
z_extraction and interval_with_gap_condition each existed in two variants).
`_time_cut` is shared — those three were identical.

Needs a parquet engine (pyarrow) at runtime; pandas raises ImportError on the
first to_parquet() without one.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from features.feature_extractors.base import FeatureExtractor


# ═════════════════════════════════════════════════════════════════════════════
# Shared helper (was defined identically in all three source extractors)
# ═════════════════════════════════════════════════════════════════════════════

def _time_cut(data: pd.DataFrame) -> float:
    """Return the earliest timestamp at which every CAN ID has appeared ≥ 2 times."""
    return float(data.loc[data.groupby("ID").cumcount() == 1, "Time"].max())

# ═══════════════════════════════════════════════════════════════════════════
# SynCAN helpers (ported from syncan_canet.py)
# ═══════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────────

_SYN_WINDOW_SIZE = 1          # original script constant
_SYN_TIME_CUTOFF = _SYN_WINDOW_SIZE + 1   # = 2 seconds  (used in _syn_prepare_dataset)

_SYN_ID_MPS: dict[str, int] = {
    "id1": 67, "id10": 22, "id2": 33, "id3": 67,
    "id4": 22, "id5": 67, "id6": 33, "id7": 67,
    "id8": 67, "id9": 33,
}
_SYN_ID_NSIG: OrderedDict[str, int] = OrderedDict([
    ("id1",  2), ("id10", 4), ("id2",  3), ("id3",  2),
    ("id4",  1), ("id5",  2), ("id6",  2), ("id7",  2),
    ("id8",  1), ("id9",  1),
])
_SYN_FIXED_IDS: list[str] = sorted(_SYN_ID_NSIG.keys())
_SYN_N_SIGS: int = sum(_SYN_ID_NSIG.values())

_SYN_TRAIN_STEMS  = ["train_1", "train_2", "train_3", "train_4"]
_SYN_VALID_STEM   = "train_valid"
_SYN_ATTACK_TYPES = ["flooding", "plateau", "continuous", "playback", "suppress"]
_SYN_TEST_STEMS   = ["test_normal"] + [f"test_{a}" for a in _SYN_ATTACK_TYPES]

_SYN_COLS = ["Label", "Time", "ID", "Signal1", "Signal2", "Signal3", "Signal4"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _syn_load_arrange_data(file_path: Path) -> pd.DataFrame:
    """
    Load a SynCAN CSV and normalise column names.
    SynCAN CSVs are ragged (each row has only as many Signal columns as
    that CAN ID transmits), so we force 7 explicit column names.
    """
    df = pd.read_csv(file_path, skiprows=1, names=_SYN_COLS, engine="python")
    df["Time"] = round(df["Time"] / 1000.0, 7)
    df.rename(columns={"Label": "Session"}, inplace=True)
    df["SessionCat"] = df["Session"].map({0: "Normal", 1: "Attack"}).fillna("Normal")
    return df


# ── CANet helpers ─────────────────────────────────────────────────────────────

def _syn_get_repeated_sequences(
    data: pd.DataFrame,
    can_id: str,
    n_sig: int,
    n_step: int,
) -> np.ndarray:
    """
    Build sliding-window sequences for one CAN ID and repeat each window
    to align it with the global message index (so every row in `data` has
    a corresponding window).

    How the alignment works:
      - Each CAN ID sends messages at its own rate (e.g. id1 sends 67 msgs/sec).
        Between two consecutive id1 messages there may be many messages from
        other IDs.  We want a window array with one entry per global row, so
        each id1 window is repeated once for every interleaved row from other IDs.
      - `n_repeats[i]` = how many global rows lie between id1_msg[i] and
        id1_msg[i+1], derived from the raw "Idx" column.

    Returns array of shape (n_global_rows_approx, n_step, n_sig).
    """
    sig_columns = [f"Signal{i}" for i in range(1, n_sig + 1)]
    df_id = data.loc[data["ID"] == can_id, ["Idx", "Session"] + sig_columns]
    np_sig = df_id[["Session"] + sig_columns].to_numpy()

    # sliding_window_view: shape (n_messages - n_step + 1, n_cols, n_step)
    # after swapaxes(1,2): shape (n_windows, n_step, n_cols)
    np_seq = np.lib.stride_tricks.sliding_window_view(np_sig, window_shape=n_step, axis=0)
    np_seq = np_seq.swapaxes(1, 2)
    np_seq = np_seq[:, :, 1:]   # drop Session column (col 0); keep only signal columns

    n_seq   = np_seq.shape[0]
    end_idx = data["Idx"].iloc[-1]
    # n_repeats[i] = global-row distance between consecutive ID messages (for alignment)
    n_repeats = np.diff(df_id["Idx"].to_list() + [end_idx])[-n_seq:]
    return np.repeat(np_seq, n_repeats, axis=0)


def _syn_prepare_dataset(
    file_path: Path,
    time_cutoff: int = _SYN_TIME_CUTOFF,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Produce per-ID window arrays and aligned time/session labels for one file.
    The first `time_cutoff` seconds are discarded (warm-up period for windows).

    Returns:
        data_dict        — {can_id: ndarray(n, _SYN_ID_MPS[id], n_sig)}
        time_and_labels  — ndarray(n, 2) with columns [Time, Session]
    """
    data = _syn_load_arrange_data(file_path)
    data = data.reset_index(names="Idx")
    time_start = data["Time"].iloc[0]

    n_rows_to_use = data.loc[data["Time"] > time_start + time_cutoff, "Time"].shape[0]
    time_and_labels = data.loc[
        data["Time"] > time_start + time_cutoff, ["Time", "Session"]
    ].to_numpy()

    data_dict: dict[str, np.ndarray] = {}
    for can_id, nsig in _SYN_ID_NSIG.items():
        seq_data = _syn_get_repeated_sequences(data, can_id, nsig, _SYN_ID_MPS[can_id])
        data_dict[can_id] = seq_data[-n_rows_to_use:].copy()

    # Align all IDs to the same (minimum) row count
    common_rows = min([n_rows_to_use] + [s.shape[0] for s in data_dict.values()])
    if common_rows <= 0:
        raise ValueError(f"No aligned CANet rows available for {file_path}")

    for can_id in list(data_dict.keys()):
        data_dict[can_id] = data_dict[can_id][-common_rows:].copy()
    time_and_labels = time_and_labels[-common_rows:].copy()
    return data_dict, time_and_labels


# ── TIL helpers ───────────────────────────────────────────────────────────────

# (_time_cut is shared — defined once above)


def _syn_get_time_intervals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-CAN-ID inter-arrival times and pivot into a wide DataFrame.
    No forward-fill or time-cutoff applied here (used only for fitting).
    """
    sub = data[["Session", "Time", "ID"]].copy()
    sub["Interval"] = sub.groupby("ID")["Time"].diff()
    base = sub[["Session", "Time"]].reset_index(drop=True)
    wide = sub.pivot_table(
        index=sub.index, columns="ID", values="Interval", aggfunc="first"
    ).reset_index(drop=True)
    return pd.concat([base, wide], axis=1)


def _syn_interval_with_gap_condition(data: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """
    For each CAN ID, compute the feature fed into TIL scoring:
        effective_interval = max(last_known_interval, elapsed_time_since_last_msg)

    Why both terms?
      - `interval_ffill`: the most recently observed inter-arrival interval for
        this ID, forward-filled to every row where that ID did not send.
      - `gap_ffill`: how long ago this ID last sent a message (current Time minus
        timestamp of its last message).
      When an ID is suppressed, `gap_ffill` grows past the expected interval, making
      the maximum spike — TIL detects this as anomalous (log-likelihood drops).

    Rows before `t_cut` (warm-up period) are dropped so every ID has appeared
    at least twice before scoring begins.
    """
    sub = data[["Session", "SessionCat", "Time", "ID"]].copy()
    t_cut = _time_cut(sub)
    sub["Interval"] = sub.groupby("ID")["Time"].diff()

    base = sub[["Session", "SessionCat", "Time"]].reset_index(drop=True)
    wide = sub.pivot_table(
        index=sub.index, columns="ID",
        values=["Time", "Interval"], aggfunc="first",
    ).reset_index(drop=True)
    merged = pd.concat([base, wide], axis=1)

    ft_dict: dict = {
        "Session":    merged["Session"].to_list(),
        "SessionCat": merged["SessionCat"].to_list(),
        "Time":       merged["Time"].to_list(),
    }
    for can_id in ids:
        # Forward-fill last seen interval across rows from other IDs
        interval_ffill = merged[("Interval", can_id)].ffill()
        # Time elapsed since the last message from this ID (grows during suppression)
        gap_ffill      = merged["Time"] - merged[("Time", can_id)].ffill()
        ft_dict[can_id] = pd.concat([interval_ffill, gap_ffill], axis=1).max(axis=1)

    ft_df = pd.DataFrame.from_dict(ft_dict)
    return ft_df.loc[ft_df["Time"] >= t_cut]


def _syn_build_norm_dist(
    train_files: list[Path],
) -> tuple[list[str], dict[str, tuple[float, float]], dict[str, list[float]]]:
    """
    Fit Gaussian distributions on inter-arrival intervals from training files.
    Matches run_til_syncan.py: _syn_get_time_intervals called with ffill=False, cut=True.

    Returns:
        target_ids  — sorted list of CAN IDs present in all training files
        norm_params — {can_id: (mu, sigma)}
        max_range   — {"itv": [global_min_loglik, global_max_loglik]}
    """
    ids_inter: set[str] = set()
    itv_dfs: list[pd.DataFrame] = []

    for data_file in train_files:
        df_data = _syn_load_arrange_data(data_file)
        unique_ids = set(df_data["ID"].unique())
        ids_inter = unique_ids.copy() if not ids_inter else ids_inter & unique_ids
        # Apply time cut (drop warm-up rows) — matches ffill=False, cut=True
        df_itv = _syn_get_time_intervals(df_data)
        t_cut  = _time_cut(df_data)
        itv_dfs.append(df_itv.loc[df_itv["Time"] >= t_cut])

    df_itv_all = pd.concat(itv_dfs, axis=0, ignore_index=True)
    gauss = pd.concat(
        [
            df_itv_all.drop(columns=["Session", "Time"]).mean().rename("mean"),
            df_itv_all.drop(columns=["Session", "Time"]).std().rename("std"),
        ],
        axis=1,
    )
    target_ids = sorted(ids_inter)

    norm_params: dict[str, tuple[float, float]] = {}
    norm_objs:   dict[str, object]               = {}
    for can_id in target_ids:
        mu    = float(gauss.loc[can_id, "mean"])
        sigma = float(gauss.loc[can_id, "std"])
        if sigma == 0 or np.isnan(sigma):
            sigma = max(abs(mu) * 0.001, 1e-9)
        norm_params[can_id] = (mu, sigma)
        norm_objs[can_id]   = norm(mu, sigma)

    # Determine [min, max] log-likelihood range across all training data
    max_range: dict[str, list[float]] = {"itv": [0.0, 0.0]}
    for data_file in train_files:
        df      = _syn_load_arrange_data(data_file)
        df_itv  = _syn_interval_with_gap_condition(df, ids=target_ids)
        ll_dict = {cid: norm_objs[cid].logpdf(df_itv[cid]) for cid in target_ids}
        ll_df   = pd.DataFrame.from_dict(ll_dict)
        cur_min = float(ll_df.sum(axis=1).min())
        cur_max = float(ll_df.sum(axis=1).max())
        if max_range["itv"][0] > cur_min:
            max_range["itv"][0] = cur_min
        if max_range["itv"][1] < cur_max:
            max_range["itv"][1] = cur_max

    return target_ids, norm_params, max_range


def _syn_z_extraction(
    data_itv: pd.DataFrame,
    ids: list[str],
    norm_params: dict[str, tuple[float, float]],
    max_range: dict[str, list[float]],
) -> pd.DataFrame:
    """
    Compute normalised log-likelihood z_itv score for each row.
    z_itv ∈ [0, 1] on the training distribution; values near 0 = anomalous.

    Formula:
        raw_ll  = Σ_ids  log N(x_id ; mu_id, sigma_id)
        z_itv   = (raw_ll − min_train) / (max_train − min_train)

    where [min_train, max_train] is the range of raw_ll observed on training
    data (stored in max_range["itv"]).  On normal traffic z_itv ≈ 1; during
    an attack (suppression, flooding, etc.) z_itv drops toward 0 or below.
    """
    # Per-ID log-likelihood under the fitted Gaussian
    ll_dict = {cid: norm(*norm_params[cid]).logpdf(data_itv[cid]) for cid in ids}
    ll_df   = pd.DataFrame.from_dict(ll_dict)
    rng     = max_range["itv"]   # [global_min_ll, global_max_ll] from training

    out = data_itv[["Session", "SessionCat", "Time"]].copy()
    # Sum log-likelihoods across all IDs, then min-max normalise to [0,1]
    out["z_itv"] = (
        (ll_df.sum(axis=1) - rng[0]) / (rng[1] - rng[0])
    ).values
    return out


# ── FeatureExtractor subclass ─────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# X-CANIDS helpers (ported from xcanids_extractor.py)
# ═══════════════════════════════════════════════════════════════════════════

# ── Attack family → label mapping (mirrors run_til_xcanids.py) ───────────────

_XC_ATTACK_MAP = {
    "fabr": "Fabrication",
    "fuzz": "Fuzzing",
    "masq": "Masquerade",
    "repl": "Replay",
    "susp": "Suspension",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _xc_load_arrange_raw(file_path: Path) -> pd.DataFrame:
    """
    Load a RAW dump parquet (dump1-5, dump6-*) and produce the canonical
    [Session, SessionCat, Time, ID] DataFrame.

    Raw parquets have columns: arbitration_id, dlc, data, label
    and a timedelta64 index named 'timestamp'.

    Ported 1-to-1 from run_til_xcanids.py load_arrange_data.
    """
    df = pd.read_parquet(file_path)

    # Derive attack family from stem (e.g. "dump6-fabr-001" → "Fabrication")
    parts  = file_path.stem.split("-")   # ["dump6", "fabr", "001"] or ["dump5"]
    family = parts[1] if len(parts) > 1 else None
    attack_label = _XC_ATTACK_MAP.get(family, None)

    # Session: 0 = normal, 1 = attack time-window
    df["Session"]    = 0
    df["SessionCat"] = "Normal"

    attack_rows = df.loc[df["label"] == 1]
    if len(attack_rows) > 0:
        t_start = attack_rows.index.min()
        t_end   = attack_rows.index.max()
        df.loc[t_start:t_end, "Session"]    = 1
        df.loc[t_start:t_end, "SessionCat"] = attack_label or "Attack"

    # Suspension special case: silent-ID window spans 480–1440 s
    if family == "susp":
        mask = (
            (480 < df.index.total_seconds()) &
            (df.index.total_seconds() <= 1440)
        )
        df.loc[mask, "Session"]    = 1
        df.loc[mask, "SessionCat"] = "Suspension"

    df.reset_index(inplace=True)
    df["Time"] = df["timestamp"].dt.total_seconds()
    df.rename(columns={"arbitration_id": "ID"}, inplace=True)
    return df[["Session", "SessionCat", "Time", "ID"]]


def _xc_load_arrange_sig107(file_path: Path) -> pd.DataFrame:
    """
    Load a pre-processed sig107 parquet and return [Session, SessionCat, Time, ID].
    Used as fallback when raw dumps are not available.

    sig107 parquets already have: MsgIndex, Session, Label, Time, ID, Signal1 …
    """
    df = pd.read_parquet(file_path)
    df.reset_index(drop=True, inplace=True)

    stem   = file_path.stem
    parts  = stem.split("-")
    family = parts[1] if len(parts) > 1 else None
    attack_label = _XC_ATTACK_MAP.get(family, None)

    df["SessionCat"] = "Normal"
    if attack_label is not None:
        df.loc[df["Session"] == 1, "SessionCat"] = attack_label

    return df[["Session", "SessionCat", "Time", "ID"]]


# ── TIL helpers (ported 1-to-1 from run_til_xcanids.py) ─────────────────────

# (_time_cut is shared — defined once above)


def _xc_get_time_intervals(
    data: pd.DataFrame, ffill: bool, cut: bool
) -> pd.DataFrame:
    """
    Compute per-CAN-ID inter-arrival times, pivot wide.
    ffill=True  → forward-fill NaN intervals.
    cut=True    → drop warm-up rows (before all IDs seen twice).
    Matches run_til_xcanids.py exactly.
    """
    data = data[["Session", "Time", "ID"]].copy()
    data["Interval"] = data.groupby("ID")["Time"].diff()
    base = data[["Session", "Time"]].reset_index(drop=True)
    wide = data.pivot_table(
        index=data.index, columns="ID", values="Interval", aggfunc="first"
    ).reset_index(drop=True)
    data_itv = pd.concat([base, wide], axis=1)
    if ffill:
        data_itv.ffill(inplace=True)
    if cut:
        t_cut = _time_cut(data)
        data_itv = data_itv.loc[data_itv["Time"] >= t_cut]
    if ffill and cut:
        assert data_itv.isna().sum().sum() == 0
    return data_itv


def _xc_interval_with_gap_condition(
    data: pd.DataFrame, ids: list[int]
) -> pd.DataFrame:
    """
    For each CAN ID compute max(forward-filled interval, elapsed gap).
    IDs are integers (matching X-CANIDS CAN arbitration IDs).
    Ported 1-to-1 from run_til_xcanids.py.
    """
    data = data[["Session", "SessionCat", "Time", "ID"]].copy()
    t_cut = _time_cut(data)
    data["Interval"] = data.groupby("ID")["Time"].diff()
    base = data[["Session", "SessionCat", "Time"]].reset_index(drop=True)
    wide = data.pivot_table(
        index=data.index, columns="ID",
        values=["Time", "Interval"], aggfunc="first",
    ).reset_index(drop=True)
    data = pd.concat([base, wide], axis=1)

    ft_dict: dict = {
        "Session":    data["Session"].to_list(),
        "SessionCat": data["SessionCat"].to_list(),
        "Time":       data["Time"].to_list(),
    }
    for can_id in ids:
        interval_ffill = data[("Interval", can_id)].ffill()
        gap_ffill      = data["Time"] - data[("Time", can_id)].ffill()
        ft_dict[can_id] = pd.concat([interval_ffill, gap_ffill], axis=1).max(axis=1)

    ft_df = pd.DataFrame.from_dict(ft_dict)
    return ft_df.loc[ft_df["Time"] >= t_cut]


def _xc_build_norm_dist(
    train_files: list[Path],
) -> tuple[list[int], dict[str, tuple[float, float]], dict[str, list[float]]]:
    """
    Fit Gaussian distributions on training inter-arrival intervals.
    Ported from run_til_xcanids.py _xc_build_norm_dist, adapted to sig107 parquets.

    Returns:
        target_ids   — sorted list of int CAN IDs common to all training files
        norm_params  — {str(can_id): (mu, sigma)}   (JSON-serialisable)
        max_range    — {"itv": [global_min_loglik, global_max_loglik]}
    """
    ids_inter: set[int] = set()
    itv_dfs: list[pd.DataFrame] = []

    for data_file in train_files:
        df_data = _xc_load_arrange_sig107(data_file)
        unique_ids = set(int(x) for x in df_data["ID"].unique())
        ids_inter  = unique_ids.copy() if not ids_inter else ids_inter & unique_ids
        # ffill=False, cut=True — matches original run_til_xcanids.py
        itv_dfs.append(_xc_get_time_intervals(df_data, ffill=False, cut=True))

    df_itv_all  = pd.concat(itv_dfs, axis=0, ignore_index=True)
    ignore_cols = ["Session", "Time"]
    gauss = pd.concat(
        [
            df_itv_all.drop(columns=ignore_cols).mean().rename("mean"),
            df_itv_all.drop(columns=ignore_cols).std().rename("std"),
        ],
        axis=1,
    )
    target_ids = sorted(ids_inter)

    norm_params: dict[str, tuple[float, float]] = {}
    norm_objs:   dict[int, object]               = {}
    for can_id in target_ids:
        mu    = float(gauss.loc[can_id, "mean"])
        sigma = float(gauss.loc[can_id, "std"])
        if sigma == 0 or np.isnan(sigma):
            sigma = float(mu) * 0.001   # matches original: norm(mean, mean*0.001)
        norm_params[str(can_id)] = (mu, sigma)
        norm_objs[can_id]        = norm(mu, sigma)

    # Compute global likelihood range across training data
    max_range: dict[str, list[float]] = {"itv": [0.0, 0.0]}
    for data_file in train_files:
        df     = _xc_load_arrange_sig107(data_file)
        df_itv = _xc_interval_with_gap_condition(df, ids=target_ids)
        ll_dict = {
            can_id: norm_objs[can_id].logpdf(df_itv[can_id])
            for can_id in target_ids
        }
        ll_df   = pd.DataFrame.from_dict(ll_dict)
        cur_min = float(ll_df.sum(axis=1).min())
        cur_max = float(ll_df.sum(axis=1).max())
        if max_range["itv"][0] > cur_min:
            max_range["itv"][0] = cur_min
        if max_range["itv"][1] < cur_max:
            max_range["itv"][1] = cur_max

    return target_ids, norm_params, max_range


def _xc_z_extraction(
    data_itv:  pd.DataFrame,
    ids:       list[int],
    norm_params: dict[str, tuple[float, float]],
    max_range: dict[str, list[float]],
) -> pd.DataFrame:
    """
    Compute normalised log-likelihood z_itv score.
    Ported 1-to-1 from run_til_xcanids.py _xc_z_extraction.
    """
    ll_dict = {
        can_id: norm(*norm_params[str(can_id)]).logpdf(data_itv[can_id])
        for can_id in ids
    }
    ll_df = pd.DataFrame.from_dict(ll_dict)
    out   = data_itv[["Session", "SessionCat", "Time"]].copy()
    rng   = max_range["itv"]
    out["z_itv"] = (
        (ll_df.sum(axis=1) - rng[0]) / (rng[1] - rng[0])
    ).values
    return out


def _xc_build_norm_dist_with_loader(
    train_files: list[Path],
    loader,
) -> tuple[list[int], dict[str, tuple[float, float]], dict[str, list[float]]]:
    """
    Wrapper around _xc_build_norm_dist that accepts an arbitrary loader function.
    Allows using _xc_load_arrange_raw (raw dumps, 62 IDs) or _xc_load_arrange_sig107
    (sig107 parquets, 35 IDs) transparently.
    """
    ids_inter: set[int] = set()
    itv_dfs: list[pd.DataFrame] = []

    for data_file in train_files:
        df_data = loader(data_file)
        unique_ids = set(int(x) for x in df_data["ID"].unique())
        ids_inter  = unique_ids.copy() if not ids_inter else ids_inter & unique_ids
        itv_dfs.append(_xc_get_time_intervals(df_data, ffill=False, cut=True))

    df_itv_all  = pd.concat(itv_dfs, axis=0, ignore_index=True)
    ignore_cols = ["Session", "Time"]
    gauss = pd.concat(
        [
            df_itv_all.drop(columns=ignore_cols).mean().rename("mean"),
            df_itv_all.drop(columns=ignore_cols).std().rename("std"),
        ],
        axis=1,
    )
    target_ids = sorted(ids_inter)

    norm_params: dict[str, tuple[float, float]] = {}
    norm_objs:   dict[int, object]               = {}
    for can_id in target_ids:
        mu    = float(gauss.loc[can_id, "mean"])
        sigma = float(gauss.loc[can_id, "std"])
        if sigma == 0 or np.isnan(sigma):
            sigma = float(mu) * 0.001
        norm_params[str(can_id)] = (mu, sigma)
        norm_objs[can_id]        = norm(mu, sigma)

    max_range: dict[str, list[float]] = {"itv": [0.0, 0.0]}
    for data_file in train_files:
        df     = loader(data_file)
        df_itv = _xc_interval_with_gap_condition(df, ids=target_ids)
        ll_dict = {
            can_id: norm_objs[can_id].logpdf(df_itv[can_id])
            for can_id in target_ids
        }
        ll_df   = pd.DataFrame.from_dict(ll_dict)
        cur_min = float(ll_df.sum(axis=1).min())
        cur_max = float(ll_df.sum(axis=1).max())
        if max_range["itv"][0] > cur_min:
            max_range["itv"][0] = cur_min
        if max_range["itv"][1] < cur_max:
            max_range["itv"][1] = cur_max

    return target_ids, norm_params, max_range


# ── FeatureExtractor subclass ─────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# ROAD helpers (ported from road_extractor.py)
# ═══════════════════════════════════════════════════════════════════════════

# ── Train / validation split (mirrors prepare_road_dataset.py) ───────────────

_ROAD_TRAIN_AMBIENT_STEMS = [
    "ambient_dyno_drive_basic_short",
    "ambient_dyno_drive_radio_infotainment",
    "ambient_dyno_drive_winter",
    "ambient_dyno_drive_extended_short",
    "ambient_dyno_reverse",
    "ambient_dyno_drive_benign_anomaly",
    "ambient_dyno_idle_radio_infotainment",
]
_ROAD_VALID_AMBIENT_STEMS = [
    "ambient_dyno_drive_extended_long",
    "ambient_dyno_drive_basic_long",
]

# ── Hyper-parameters (mirrors prepare_road_dataset.py) ───────────────────────

_ROAD_MIN_RATE          = 5.0    # minimum msgs/sec for an ID to be included
_ROAD_MAX_WINDOW        = 200    # cap on CANet sliding-window width
_ROAD_MIN_COVERAGE_FRAC = 0.6    # fraction of training files an ID must appear in
_ROAD_MAX_SIGNALS  = 22     # maximum decoded signals per CAN ID in ROAD CSVs

# ── Focused IDs for TIL (mirrors run_til_road.py) ────────────────────────────

_ROAD_FOCUSED_IDS: list[int] = [208, 1255, 1760]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers ported from prepare_road_dataset.py
# ═══════════════════════════════════════════════════════════════════════════════

def _road_read_csv(file_path: Path) -> pd.DataFrame:
    """Load a ROAD signal_extractions CSV and rename Signal_N_of_ID → SignalN."""
    df = pd.read_csv(file_path)
    rename = {
        col: f"Signal{col.split('Signal_')[1].split('_of_ID')[0]}"
        for col in df.columns
        if col.startswith("Signal_") and "_of_ID" in col
    }
    df = df.rename(columns=rename)
    df["ID"] = df["ID"].astype(int)
    return df


def _road_assign_session(df: pd.DataFrame) -> pd.DataFrame:
    """Add Session (0/1) and SessionCat columns based on Label."""
    df = df.copy()
    df["Session"]    = 0
    df["SessionCat"] = "Normal"
    if "Label" in df.columns:
        attack_rows = df["Label"] == 1
        if attack_rows.any():
            t_start = df.loc[attack_rows, "Time"].min()
            t_end   = df.loc[attack_rows, "Time"].max()
            mask = (df["Time"] >= t_start) & (df["Time"] <= t_end)
            df.loc[mask, "Session"]    = 1
            df.loc[mask, "SessionCat"] = "Attack"
    df["Session"] = df["Session"].astype(int)
    if "Label" not in df.columns:
        df["Label"] = 0
    df["Label"] = df["Label"].astype(int)
    return df


def _road_get_n_signals(df: pd.DataFrame, can_id: int) -> int:
    """Count how many SignalN columns have non-NaN values for this CAN ID."""
    rows = df[df["ID"] == can_id]
    if rows.empty:
        return 0
    for i in range(1, _ROAD_MAX_SIGNALS + 2):
        col = f"Signal{i}"
        if col not in rows.columns or rows[col].isna().all():
            return max(i - 1, 1)
    return _ROAD_MAX_SIGNALS


def _road_analyze_training_data(
    train_files: list[Path],
    min_rate: float,
    min_coverage_frac: float,
) -> tuple[list[int], dict[int, int], dict[int, int]]:
    """
    Scan training files and return:
        final_ids — sorted list of CAN IDs to model
        id_nsig   — {can_id: number_of_signals}
        id_mps    — {can_id: window_size_in_messages}
    Ported 1-to-1 from prepare_road_dataset.py.
    """
    id_count:     dict[int, int]         = defaultdict(int)
    id_nsig_list: dict[int, list[int]]   = defaultdict(list)
    id_rate_list: dict[int, list[float]] = defaultdict(list)

    for fp in tqdm(train_files, desc="  Scanning training files"):
        df       = _road_read_csv(fp)
        duration = df["Time"].max() - df["Time"].min()
        if duration <= 0:
            continue
        for can_id in df["ID"].unique():
            id_count[int(can_id)] += 1
            nsig = _road_get_n_signals(df, int(can_id))
            if nsig > 0:
                id_nsig_list[int(can_id)].append(nsig)
            rate = float((df["ID"] == can_id).sum()) / duration
            id_rate_list[int(can_id)].append(rate)

    min_files  = max(1, int(len(train_files) * min_coverage_frac))
    final_ids: list[int]       = []
    id_nsig:   dict[int, int]  = {}
    id_mps:    dict[int, int]  = {}

    for can_id, count in id_count.items():
        if count < min_files:
            continue
        median_rate = float(np.median(id_rate_list[can_id]))
        if median_rate < min_rate:
            continue
        nsigs       = id_nsig_list[can_id]
        median_nsig = int(np.median(nsigs)) if nsigs else 1
        final_ids.append(can_id)
        id_nsig[can_id] = max(1, median_nsig)
        id_mps[can_id]  = min(_ROAD_MAX_WINDOW, max(1, int(round(median_rate))))

    return sorted(final_ids), id_nsig, id_mps


def _road_compute_norm_stats(
    train_files: list[Path],
    ids:         list[int],
    id_nsig:     dict[int, int],
) -> dict[str, dict[str, float]]:
    """
    Compute per-ID-per-signal min/max over all training files.
    Ported 1-to-1 from prepare_road_dataset.py.
    """
    sig_min: dict[str, float] = {}
    sig_max: dict[str, float] = {}

    for fp in tqdm(train_files, desc="  Computing norm stats"):
        df = _road_read_csv(fp)
        for can_id in ids:
            rows = df[df["ID"] == can_id]
            if rows.empty:
                continue
            for i in range(1, id_nsig[can_id] + 1):
                col = f"Signal{i}"
                key = f"{can_id}_{i}"
                if col not in rows.columns:
                    continue
                vals = rows[col].dropna()
                if vals.empty:
                    continue
                v_min = float(vals.min())
                v_max = float(vals.max())
                if key not in sig_min:
                    sig_min[key] = v_min
                    sig_max[key] = v_max
                else:
                    sig_min[key] = min(sig_min[key], v_min)
                    sig_max[key] = max(sig_max[key], v_max)

    return {k: {"min": sig_min[k], "max": sig_max[k]} for k in sig_min}


def _road_convert_csv_to_canet_parquet(
    file_path:  Path,
    ids:        list[int],
    id_nsig:    dict[int, int],
    norm_stats: dict[str, dict[str, float]],
    output_path: Path,
) -> int:
    """
    Convert a ROAD signal CSV → CANet-format parquet.
    Output columns: MsgIndex, Session, Label, Time, ID (str), Signal1 … SignalN
    Normalization: (val − min) / (max − min) clipped to [0, 1].
    Ported 1-to-1 from prepare_road_dataset.py.
    """
    df = _road_read_csv(file_path)
    df = _road_assign_session(df)
    df = df.reset_index().rename(columns={"index": "MsgIndex"})

    parts: list[pd.DataFrame] = []
    for can_id in ids:
        n_sig = id_nsig[can_id]
        rows  = df[df["ID"] == can_id].copy()
        if rows.empty:
            continue

        out      = rows[["MsgIndex", "Session", "Label", "Time"]].copy()
        out["ID"] = str(can_id)

        for i in range(1, n_sig + 1):
            col   = f"Signal{i}"
            key   = f"{can_id}_{i}"
            stats = norm_stats.get(key, {})
            mn    = stats.get("min")
            mx    = stats.get("max")

            if col in rows.columns:
                vals = rows[col].fillna(0.0)
                if mn is not None and mx is not None and mx > mn:
                    vals = ((vals - mn) / (mx - mn)).clip(0.0, 1.0)
            else:
                vals = pd.Series(0.0, index=rows.index)

            out[f"Signal{i}"] = vals.values

        parts.append(out)

    if not parts:
        print(f"  [WARN] No matching IDs found in {file_path.name}")
        return 0

    df_out = (
        pd.concat(parts, axis=0)
        .sort_values("MsgIndex", ignore_index=True)
        .astype({"Session": int, "Label": int})
    )
    df_out.to_parquet(output_path)
    return len(df_out)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers ported from run_til_road.py
# ═══════════════════════════════════════════════════════════════════════════════

def _road_load_arrange_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a ROAD CSV for TIL scoring (only Label, Time, ID needed).
    Adds Session and SessionCat columns.
    Ported 1-to-1 from run_til_road.py.
    """
    df = pd.read_csv(file_path, usecols=lambda c: c in {"Label", "Time", "ID"})
    df["ID"] = df["ID"].astype(int)
    if "Label" not in df.columns:
        df["Label"] = 0
    df["Label"]      = df["Label"].astype(int)
    df["Session"]    = 0
    df["SessionCat"] = "Normal"
    attack_rows = df["Label"] == 1
    if attack_rows.any():
        t_start = df.loc[attack_rows, "Time"].min()
        t_end   = df.loc[attack_rows, "Time"].max()
        mask    = (df["Time"] >= t_start) & (df["Time"] <= t_end)
        df.loc[mask, "Session"]    = 1
        df.loc[mask, "SessionCat"] = "Attack"
    df["Session"] = df["Session"].astype(int)
    return df[["Session", "SessionCat", "Time", "ID"]]


# (_time_cut is shared — defined once above)


def _road_get_time_intervals(
    data: pd.DataFrame, ffill: bool, cut: bool
) -> pd.DataFrame:
    """
    Compute per-CAN-ID inter-arrival times, pivot wide.
    Ported 1-to-1 from run_til_road.py get_time_intervals.
    """
    data = data[["Session", "Time", "ID"]].copy()
    data["Interval"] = data.groupby("ID")["Time"].diff()
    base = data[["Session", "Time"]].reset_index(drop=True)
    wide = data.pivot_table(
        index=data.index, columns="ID", values="Interval", aggfunc="first"
    ).reset_index(drop=True)
    df_itv = pd.concat([base, wide], axis=1)
    if ffill:
        df_itv.ffill(inplace=True)
    if cut:
        t_cut  = _time_cut(data)
        df_itv = df_itv.loc[df_itv["Time"] >= t_cut]
    if ffill and cut and df_itv.isna().sum().sum() > 0:
        df_itv = df_itv.ffill()
    return df_itv


def _road_interval_with_gap_condition(
    data: pd.DataFrame, ids: list[int]
) -> pd.DataFrame:
    """
    For each CAN ID compute max(forward-filled interval, elapsed gap).
    Ported 1-to-1 from run_til_road.py interval_with_gap_condition.
    """
    data = data[["Session", "SessionCat", "Time", "ID"]].copy()
    t_cut = _time_cut(data)
    data["Interval"] = data.groupby("ID")["Time"].diff()
    base = data[["Session", "SessionCat", "Time"]].reset_index(drop=True)
    wide = data.pivot_table(
        index=data.index, columns="ID",
        values=["Time", "Interval"], aggfunc="first",
    ).reset_index(drop=True)
    data_w = pd.concat([base, wide], axis=1)

    ft: dict = {
        "Session":    data_w["Session"].to_list(),
        "SessionCat": data_w["SessionCat"].to_list(),
        "Time":       data_w["Time"].to_list(),
    }
    for cid in ids:
        if ("Interval", cid) not in data_w.columns or ("Time", cid) not in data_w.columns:
            continue
        interval_ff = data_w[("Interval", cid)].ffill()
        gap_ff      = data_w["Time"] - data_w[("Time", cid)].ffill()
        ft[cid] = pd.concat([interval_ff, gap_ff], axis=1).max(axis=1)

    ft_df = pd.DataFrame.from_dict(ft)
    return ft_df.loc[ft_df["Time"] >= t_cut]


def _road_build_norm_dist_focused(
    train_files: list[Path],
) -> tuple[
    list[int],
    dict[str, tuple[float, float]],
    dict[str, list[float]],
    dict[str, list[float]],
]:
    """
    Fit global + focused Gaussian distributions on training intervals.
    Ported 1-to-1 from run_til_road.py _road_build_norm_dist_focused.

    Returns:
        target_ids    — sorted int CAN IDs common to all training files
        norm_params   — {str(can_id): (mu, sigma)}
        max_range     — {"itv": [global_min, global_max]}
        focused_range — {str(focused_id): [min, max]}
    """
    ids_inter: set[int] = set()
    itv_dfs: list[pd.DataFrame] = []

    for fp in train_files:
        df = _road_load_arrange_csv(fp)
        unique_ids = set(int(x) for x in df["ID"].unique())
        ids_inter  = unique_ids.copy() if not ids_inter else ids_inter & unique_ids
        itv_dfs.append(_road_get_time_intervals(df, ffill=False, cut=True))

    df_itv_all  = pd.concat(itv_dfs, axis=0, ignore_index=True)
    ignore_cols = {"Session", "Time"}
    gauss = pd.concat(
        [
            df_itv_all.drop(columns=list(ignore_cols)).mean().rename("mean"),
            df_itv_all.drop(columns=list(ignore_cols)).std().rename("std"),
        ],
        axis=1,
    )
    target_ids = sorted(ids_inter)

    norm_params: dict[str, tuple[float, float]] = {}
    norm_objs:   dict[int, object]               = {}
    for cid in target_ids:
        mu    = float(gauss.loc[cid, "mean"])
        sigma = float(gauss.loc[cid, "std"])
        if np.isnan(sigma) or sigma == 0:
            sigma = max(abs(mu) * 0.001, 1e-9)
        norm_params[str(cid)] = (mu, sigma)
        norm_objs[cid]        = norm(mu, sigma)

    # Global likelihood range
    max_range: dict[str, list[float]] = {"itv": [float("+inf"), float("-inf")]}
    # Per-focused-ID likelihood range
    focused_range: dict[str, list[float]] = {
        str(cid): [float("+inf"), float("-inf")] for cid in _ROAD_FOCUSED_IDS
    }

    for fp in train_files:
        df     = _road_load_arrange_csv(fp)
        df_itv = _road_interval_with_gap_condition(df, ids=target_ids)

        lk_cols: dict[int, pd.Series] = {}
        for cid in target_ids:
            if cid not in df_itv.columns or cid not in norm_objs:
                continue
            lk_cols[cid] = norm_objs[cid].logpdf(df_itv[cid])

        if lk_cols:
            row_sum = pd.DataFrame(lk_cols).sum(axis=1)
            c_min, c_max = float(row_sum.min()), float(row_sum.max())
            if max_range["itv"][0] > c_min:
                max_range["itv"][0] = c_min
            if max_range["itv"][1] < c_max:
                max_range["itv"][1] = c_max

        for cid in _ROAD_FOCUSED_IDS:
            if cid not in df_itv.columns or cid not in norm_objs:
                continue
            lk_id = norm_objs[cid].logpdf(df_itv[cid])
            c_min_id = float(lk_id.min())
            c_max_id = float(lk_id.max())
            if focused_range[str(cid)][0] > c_min_id:
                focused_range[str(cid)][0] = c_min_id
            if focused_range[str(cid)][1] < c_max_id:
                focused_range[str(cid)][1] = c_max_id

    return target_ids, norm_params, max_range, focused_range


def _road_z_extraction_focused(
    df_itv:        pd.DataFrame,
    ids:           list[int],
    norm_params:   dict[str, tuple[float, float]],
    max_range:     dict[str, list[float]],
    focused_range: dict[str, list[float]],
) -> pd.DataFrame:
    """
    Compute global z_itv + per-focused-ID z_itv_{cid} scores.
    Ported 1-to-1 from run_til_road.py _road_z_extraction_focused.
    Output columns: Session, SessionCat, Time, z_itv,
                    z_itv_208, z_itv_1255, z_itv_1760
    """
    lk_cols: dict[int, pd.Series] = {}
    for cid in ids:
        if cid not in df_itv.columns:
            continue
        lk_cols[cid] = norm(*norm_params[str(cid)]).logpdf(df_itv[cid])

    out = df_itv[["Session", "SessionCat", "Time"]].copy()

    if lk_cols:
        lk_df   = pd.DataFrame(lk_cols)
        rng     = max_range["itv"][1] - max_range["itv"][0]
        row_sum = lk_df.sum(axis=1).to_numpy()
        out["z_itv"] = (
            (row_sum - max_range["itv"][0]) / rng if rng != 0 else 0.5
        )
    else:
        out["z_itv"] = 0.5

    for cid in _ROAD_FOCUSED_IDS:
        col_name = f"z_itv_{cid}"
        if cid not in lk_cols:
            out[col_name] = np.nan
            continue
        lk_id   = np.asarray(lk_cols[cid])
        rng_id  = focused_range[str(cid)][1] - focused_range[str(cid)][0]
        out[col_name] = (
            (lk_id - focused_range[str(cid)][0]) / rng_id if rng_id != 0 else 0.5
        )

    return out


# ── FeatureExtractor subclass ─────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════════
# The extractor
# ═════════════════════════════════════════════════════════════════════════════

class CanivalExtractor(FeatureExtractor):
    """Prepares TIL + CANet features for SynCAN, X-CANIDS or ROAD.

    Work happens in __init__ (via extract()), matching how every other
    extractor in this pipeline behaves -- src/get_extractor.py only
    constructs the class, it never calls a method on it.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.extract()

    def extract(self):
        name = (self.dataset_name or "").strip()
        print(f"\n{'=' * 60}")
        print("  CanivalExtractor — Feature Extraction")
        print(f"  Dataset: {name}")
        print(f"{'=' * 60}")
        if name == "SynCAN":
            return self._extract_syncan()
        if name == "X-CANIDS":
            return self._extract_xcanids()
        if name == "ROAD":
            return self._extract_road()
        raise NotImplementedError(
            f"CanivalExtractor: unsupported dataset_name {name!r}. "
            "CANival supports SynCAN, X-CANIDS and ROAD."
        )

    def _extract_syncan(self):
        mod_dir  = Path(self.dataset_path)
        feat_dir = Path(self.features_path)

        canet_dir = feat_dir / "canet"
        til_dir   = feat_dir / "til"
        canet_dir.mkdir(parents=True, exist_ok=True)
        til_dir.mkdir(parents=True, exist_ok=True)

        # ── Gather file paths ────────────────────────────────────────────────
        train_files = [mod_dir / f"{s}.csv" for s in _SYN_TRAIN_STEMS]
        valid_file  = mod_dir / f"{_SYN_VALID_STEM}.csv"
        test_files  = [mod_dir / f"{s}.csv" for s in _SYN_TEST_STEMS]
        all_files   = train_files + [valid_file] + test_files

        missing = [str(f) for f in all_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"SynCAN: missing files in modified_dataset/: {missing}"
            )

        # ── CANet: sliding-window sequences for every file ───────────────────
        print("  [SynCAN] Extracting CANet window sequences …")
        for file_path in all_files:
            stem     = file_path.stem
            out_path = canet_dir / f"syncan_canet_{stem}.npz"
            if out_path.exists():
                print(f"    [skip] {out_path.name}")
                continue

            data_dict, time_and_labels = _syn_prepare_dataset(file_path)
            # y = last step of each window, all IDs concatenated
            y = np.concatenate(
                [data_dict[can_id][:, -1, :] for can_id in _SYN_FIXED_IDS], axis=1
            )
            np.savez_compressed(
                out_path,
                **{can_id: data_dict[can_id] for can_id in _SYN_FIXED_IDS},
                y=y,
                time_and_labels=time_and_labels,
            )
            print(f"    Saved {out_path.name}  rows={len(y)}")

        # ── TIL: fit Gaussian distributions from training files ──────────────
        print("  [SynCAN] Fitting TIL normal distributions …")
        norm_dist_path = til_dir / "syncan_til_norm_dist.json"

        if norm_dist_path.exists():
            print(f"    [skip] {norm_dist_path.name} (loading existing)")
            with open(norm_dist_path) as fh:
                nd = json.load(fh)
            target_ids  = nd["target_ids"]
            norm_params = {k: tuple(v) for k, v in nd["norm_params"].items()}
            max_range   = nd["max_range"]
        else:
            target_ids, norm_params, max_range = _syn_build_norm_dist(train_files)
            with open(norm_dist_path, "w") as fh:
                json.dump(
                    {
                        "target_ids":  target_ids,
                        "norm_params": norm_params,   # {id: [mu, sigma]}
                        "max_range":   max_range,
                    },
                    fh, indent=2,
                )
            print(f"    Saved {norm_dist_path.name}  ids={len(target_ids)}")

        # ── TIL: compute z_itv for valid + test files ────────────────────────
        print("  [SynCAN] Computing TIL z-scores …")
        for file_path in [valid_file] + test_files:
            stem     = file_path.stem
            out_path = til_dir / f"syncan_til_{stem}.parquet"
            if out_path.exists():
                print(f"    [skip] {out_path.name}")
                continue

            df     = _syn_load_arrange_data(file_path)
            df_itv = _syn_interval_with_gap_condition(df, ids=target_ids)
            df_z   = _syn_z_extraction(
                df_itv,
                ids=target_ids,
                norm_params=norm_params,
                max_range=max_range,
            )
            df_z.to_parquet(out_path, index=False)
            print(f"    Saved {out_path.name}  rows={len(df_z)}")

        # ── Summary ──────────────────────────────────────────────────────────
        n_canet = len(list(canet_dir.glob("*.npz")))
        n_til   = len(list(til_dir.glob("*.parquet")))
        print("  [SynCAN] Feature extraction complete.")
        print(f"    CANet .npz files  : {n_canet}")
        print(f"    TIL  .parquet files: {n_til}")

    def _extract_xcanids(self):
        mod_dir  = Path(self.dataset_path)
        feat_dir = Path(self.features_path)
        # original_dataset lives one level above modified_dataset
        orig_dir = mod_dir.parent / "original_dataset"
        til_dir  = feat_dir / "til"
        til_dir.mkdir(parents=True, exist_ok=True)

        # ── Locate sig107 parquets (CANet) ───────────────────────────────────
        train_sig107 = sorted(mod_dir.glob("sig107_dump[1-4].parquet"))
        valid_sig107 = sorted(mod_dir.glob("sig107_dump5.parquet"))
        test_sig107  = sorted(mod_dir.glob("sig107_dump6-*.parquet"))

        if not train_sig107:
            raise FileNotFoundError(
                "X-CANIDS: no sig107_dump[1-4].parquet found in modified_dataset/. "
                "Run preprocess_dataset.py first."
            )
        if not valid_sig107:
            raise FileNotFoundError(
                "X-CANIDS: sig107_dump5.parquet not found in modified_dataset/."
            )

        print(f"  [X-CANIDS] CANet sig107  train: {len(train_sig107)}")
        print(f"  [X-CANIDS] CANet sig107  valid: {len(valid_sig107)}")
        print(f"  [X-CANIDS] CANet sig107  test : {len(test_sig107)}")
        print("  [X-CANIDS] (CANet: sig107 parquets used directly at inference time)")

        # ── Locate raw dump parquets (TIL) ───────────────────────────────────
        # TIL uses raw dumps (all 62 CAN IDs) to match run_til_xcanids.py exactly.
        # Raw dumps live in original_dataset/ (copied there by preprocess_dataset.py).
        # Fall back to sig107 if raw dumps are unavailable.
        raw_train = sorted(orig_dir.glob("dump[1-4].parquet")) if orig_dir.exists() else []
        raw_valid = sorted(orig_dir.glob("dump5.parquet"))     if orig_dir.exists() else []
        raw_test  = sorted(orig_dir.glob("dump6-*.parquet"))   if orig_dir.exists() else []

        if raw_train:
            til_train = raw_train
            til_valid = raw_valid
            til_test  = raw_test
            til_loader = _xc_load_arrange_raw
            print(f"  [X-CANIDS] TIL raw dump train: {len(til_train)}")
            print(f"  [X-CANIDS] TIL raw dump valid: {len(til_valid)}")
            print(f"  [X-CANIDS] TIL raw dump test : {len(til_test)}")
        else:
            # Fallback: use sig107 parquets (35 IDs instead of 62)
            til_train  = train_sig107
            til_valid  = valid_sig107
            til_test   = test_sig107
            til_loader = _xc_load_arrange_sig107
            print("  [X-CANIDS] TIL: raw dumps not found — using sig107 parquets (35 IDs)")

        # ── TIL: fit Gaussian distributions from training parquets ───────────
        print("  [X-CANIDS] Fitting TIL normal distributions …")
        norm_dist_path = til_dir / "xcanids_til_norm_dist.json"

        if norm_dist_path.exists():
            print(f"    [skip] {norm_dist_path.name} (loading existing)")
            with open(norm_dist_path) as fh:
                nd = json.load(fh)
            target_ids  = [int(x) for x in nd["target_ids"]]
            norm_params = {k: tuple(v) for k, v in nd["norm_params"].items()}
            max_range   = nd["max_range"]
        else:
            # _xc_build_norm_dist calls til_loader via _xc_load_arrange_sig107 inside it.
            # We need to temporarily patch _xc_load_arrange_sig107 to use the right loader.
            # Simpler: call _xc_build_norm_dist which uses _xc_load_arrange_sig107 internally —
            # but now we want it to use til_loader. Since _xc_build_norm_dist calls
            # _xc_load_arrange_sig107 by name, we pass the files and loader explicitly here.
            target_ids, norm_params, max_range = _xc_build_norm_dist_with_loader(
                til_train, til_loader
            )
            with open(norm_dist_path, "w") as fh:
                json.dump(
                    {
                        "target_ids":  target_ids,
                        "norm_params": norm_params,
                        "max_range":   max_range,
                    },
                    fh, indent=2,
                )
            print(f"    Saved {norm_dist_path.name}  ids={len(target_ids)}")

        # ── TIL: compute z_itv for valid + test parquets ─────────────────────
        # Output stem uses the raw dump stem (e.g. "dump5") or sig107 stem.
        # The _test_xcanids in til.py looks for xcanids_til_dump5.parquet
        # (raw) or xcanids_til_sig107_dump5.parquet (sig107 fallback).
        print("  [X-CANIDS] Computing TIL z-scores …")
        for file_path in til_valid + til_test:
            stem     = file_path.stem   # "dump5" or "dump6-fabr-001" (raw)
            out_path = til_dir / f"xcanids_til_{stem}.parquet"
            if out_path.exists():
                print(f"    [skip] {out_path.name}")
                continue

            df     = til_loader(file_path)
            df_itv = _xc_interval_with_gap_condition(df, ids=target_ids)
            df_z   = _xc_z_extraction(
                df_itv,
                ids=target_ids,
                norm_params=norm_params,
                max_range=max_range,
            )
            df_z.to_parquet(out_path, index=False)
            print(f"    Saved {out_path.name}  rows={len(df_z)}")

        # ── Summary ──────────────────────────────────────────────────────────
        n_til = len(list(til_dir.glob("*.parquet")))
        print("  [X-CANIDS] Feature extraction complete.")
        print(f"    TIL parquet files : {n_til}")

    def _extract_road(self):
        mod_dir  = Path(self.dataset_path)
        feat_dir = Path(self.features_path)

        canet_dir = feat_dir / "canet"
        til_dir   = feat_dir / "til"
        canet_dir.mkdir(parents=True, exist_ok=True)
        til_dir.mkdir(parents=True, exist_ok=True)

        ambient_dir = mod_dir / "ambient"
        attacks_dir = mod_dir / "attacks"

        if not ambient_dir.exists():
            raise FileNotFoundError(
                f"ROAD: {ambient_dir} not found. Run preprocess_dataset.py first."
            )
        if not attacks_dir.exists():
            raise FileNotFoundError(
                f"ROAD: {attacks_dir} not found. Run preprocess_dataset.py first."
            )

        # ── Gather CSV files ─────────────────────────────────────────────────
        train_files = [
            ambient_dir / f"{s}.csv"
            for s in _ROAD_TRAIN_AMBIENT_STEMS
            if (ambient_dir / f"{s}.csv").exists()
        ]
        valid_files = [
            ambient_dir / f"{s}.csv"
            for s in _ROAD_VALID_AMBIENT_STEMS
            if (ambient_dir / f"{s}.csv").exists()
        ]
        test_files = sorted(attacks_dir.glob("*.csv"))

        print(f"  [ROAD] Train ambient CSVs : {len(train_files)}")
        print(f"  [ROAD] Valid ambient CSVs : {len(valid_files)}")
        print(f"  [ROAD] Attack CSVs        : {len(test_files)}")

        if not train_files:
            raise FileNotFoundError(
                "ROAD: no training ambient CSVs found in modified_dataset/ambient/."
            )

        # ── Step 1 & 2: Load or build road_config ────────────────────────────
        config_path = feat_dir / "road_config.json"
        if config_path.exists():
            print(f"  [ROAD] Loading existing config: {config_path.name}")
            with open(config_path) as fh:
                cfg = json.load(fh)
            final_ids  = [int(x) for x in cfg["final_ids"]]
            id_nsig    = {int(k): v for k, v in cfg["id_nsig"].items()}
            id_mps     = {int(k): v for k, v in cfg["id_mps"].items()}
            norm_stats = cfg["norm_stats"]
        else:
            print("  [ROAD] Analysing training data to select CAN IDs …")
            final_ids, id_nsig, id_mps = _road_analyze_training_data(
                train_files,
                min_rate=_ROAD_MIN_RATE,
                min_coverage_frac=_ROAD_MIN_COVERAGE_FRAC,
            )
            n_total = sum(id_nsig.values())
            print(f"    Selected IDs : {len(final_ids)},  Total signals : {n_total}")

            print("  [ROAD] Computing per-signal normalization stats …")
            norm_stats = _road_compute_norm_stats(train_files, final_ids, id_nsig)

            cfg = {
                "dataset":       "ROAD",
                "final_ids":     final_ids,
                "id_nsig":       {str(k): v for k, v in id_nsig.items()},
                "id_mps":        {str(k): v for k, v in id_mps.items()},
                "n_sigs_total":  n_total,
                "norm_stats":    norm_stats,
                "train_stems":   _ROAD_TRAIN_AMBIENT_STEMS,
                "valid_stems":   _ROAD_VALID_AMBIENT_STEMS,
            }
            with open(config_path, "w") as fh:
                json.dump(cfg, fh, indent=2)
            print(f"    Config saved → {config_path.name}")

        # ── Step 3: Convert all CSVs → CANet parquets ────────────────────────
        print("  [ROAD] Converting training CSVs → CANet parquets …")
        for fp in tqdm(train_files, desc="  train"):
            out = canet_dir / f"train_{fp.stem}.parquet"
            if out.exists():
                continue
            n = _road_convert_csv_to_canet_parquet(fp, final_ids, id_nsig, norm_stats, out)
            tqdm.write(f"    {fp.name} → {n:,} rows → {out.name}")

        print("  [ROAD] Converting validation CSVs → CANet parquets …")
        for fp in tqdm(valid_files, desc="  valid"):
            out = canet_dir / f"valid_{fp.stem}.parquet"
            if out.exists():
                continue
            n = _road_convert_csv_to_canet_parquet(fp, final_ids, id_nsig, norm_stats, out)
            tqdm.write(f"    {fp.name} → {n:,} rows → {out.name}")

        print("  [ROAD] Converting attack CSVs → CANet parquets …")
        for fp in tqdm(test_files, desc="  test"):
            out = canet_dir / f"test_{fp.stem}.parquet"
            if out.exists():
                continue
            n = _road_convert_csv_to_canet_parquet(fp, final_ids, id_nsig, norm_stats, out)
            tqdm.write(f"    {fp.name} → {n:,} rows → {out.name}")

        # ── Step 4: Build focused TIL distributions ───────────────────────────
        print("  [ROAD] Fitting TIL distributions (global + focused) …")
        norm_dist_path = til_dir / "road_til_norm_dist.json"

        if norm_dist_path.exists():
            print(f"    [skip] {norm_dist_path.name} (loading existing)")
            with open(norm_dist_path) as fh:
                nd = json.load(fh)
            target_ids    = [int(x) for x in nd["target_ids"]]
            norm_params   = {k: tuple(v) for k, v in nd["norm_params"].items()}
            max_range     = nd["max_range"]
            focused_range = {k: list(v) for k, v in nd["focused_range"].items()}
        else:
            target_ids, norm_params, max_range, focused_range = _road_build_norm_dist_focused(
                train_files
            )
            print(f"    Global  range : {max_range['itv']}")
            for cid in _ROAD_FOCUSED_IDS:
                print(f"    ID {cid:4d} range : {focused_range[str(cid)]}")

            with open(norm_dist_path, "w") as fh:
                json.dump(
                    {
                        "target_ids":    target_ids,
                        "norm_params":   norm_params,   # {str(id): [mu, sigma]}
                        "max_range":     max_range,
                        "focused_range": focused_range, # {str(id): [min, max]}
                    },
                    fh, indent=2,
                )
            print(f"    Saved {norm_dist_path.name}  ids={len(target_ids)}")

        # ── Step 5: Compute z-scores for valid + test ─────────────────────────
        print("  [ROAD] Computing TIL z-scores (global + focused) …")
        eval_files = valid_files + list(test_files)
        for fp in tqdm(eval_files, desc="  scoring"):
            stem     = fp.stem
            out_path = til_dir / f"road_til_{stem}.parquet"
            if out_path.exists():
                tqdm.write(f"    [skip] {out_path.name}")
                continue

            df     = _road_load_arrange_csv(fp)
            df_itv = _road_interval_with_gap_condition(df, ids=target_ids)
            df_z   = _road_z_extraction_focused(
                df_itv,
                ids=target_ids,
                norm_params=norm_params,
                max_range=max_range,
                focused_range=focused_range,
            )
            df_z.to_parquet(out_path, index=False)
            tqdm.write(f"    Saved {out_path.name}  rows={len(df_z)}")

        # ── Summary ──────────────────────────────────────────────────────────
        n_canet = len(list(canet_dir.glob("*.parquet")))
        n_til   = len(list(til_dir.glob("*.parquet")))
        print("  [ROAD] Feature extraction complete.")
        print(f"    CANet parquets   : {n_canet}")
        print(f"    TIL   parquets   : {n_til}")
        print(f"    Config           : {config_path.name}")
