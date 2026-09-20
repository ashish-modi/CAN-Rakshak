"""Run full statistical analysis + plots for all CAN IDS datasets.

For every CSV file found, outputs to:
  datasets/<Dataset>/analysis/<file_stem>/
    statistics.txt        ← stats report
    canid_distribution.png
    message_rate.png
    class_distribution.png
    canid_vs_time.png
    canid_vs_time_zoom.png
    canid_periodicity.png
    payload_entropy_clean.png
    byte_correlation.png
    label_vs_time.png
"""

import os
import sys

# Allow running from the repo root or from src/
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for _p in (_here, _root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_analysis import (
    convert_payload_to_int,
    basic_statistics,
    save_statistics,
    plot_can_id_distribution,
    plot_message_rate,
    plot_class_distribution,
    plot_canid_vs_time,
    plot_payload_entropy,
    plot_byte_correlation,
    plot_canid_periodicity,
    plot_label_vs_time,
    plot_label_vs_time_windows,
)

BASE = os.path.join(_root, "datasets")

DATASETS = [
    ("my_dataset/MIRGU",           "modified_dataset", False),
    # ("CarHackingDataset",           "modified_dataset", False),
    # ("CANIntrusionDataset",         "modified_dataset", False),
    # ("Syn_MIRGU_spoof",             "modified_dataset", True),
    # ("CARLA",                       "modified_dataset", False),
]

COLS = ["timestamp", "can_id", "dlc",
        "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "flag"]

# ─────────────────────────────────────────────────────────────
def load_csv(path, has_header):
    if has_header:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        # normalise column names to match expected schema
        rename = {"label": "flag", "can_id": "can_id", "dlc": "dlc"}
        df = df.rename(columns=rename)
    else:
        df = pd.read_csv(path, header=None, names=COLS, low_memory=False)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["dlc"]       = pd.to_numeric(df["dlc"],       errors="coerce")
    df["can_id"]    = df["can_id"].astype(str).str.strip().str.lower()
    df["flag"]      = df["flag"].astype(str).str.strip()
    return df


def attack_type_from_name(stem):
    """Best-effort attack label from the file stem."""
    s = stem.lower()
    if "dos"            in s: return "DoS Attack"
    if "fuzzy"          in s: return "Fuzzy Attack"
    if "gear"           in s: return "Gear Spoofing"
    if "impersonation"  in s: return "Impersonation Attack"
    if "spoof"          in s: return "Spoofing Attack"
    if "free" in s or "normal" in s or "benign" in s: return "Normal (no attack)"
    return "Unknown"


# ─────────────────────────────────────────────────────────────
def run_dataset(ds_folder, modified_sub, has_header):
    modified_dir = os.path.join(BASE, ds_folder, modified_sub)
    if not os.path.isdir(modified_dir):
        print(f"  [SKIP] {modified_dir} — directory not found")
        return

    csv_files = sorted(f for f in os.listdir(modified_dir) if f.endswith(".csv"))
    if not csv_files:
        print(f"  [SKIP] {modified_dir} — no CSV files")
        return

    print(f"\n{'='*70}")
    print(f"  Dataset : {ds_folder}  ({len(csv_files)} files)")
    print(f"  Source  : {modified_dir}")
    print(f"{'='*70}")

    for csv_file in csv_files:
        file_path = os.path.join(modified_dir, csv_file)
        stem      = csv_file.rsplit(".", 1)[0]
        atk_type  = attack_type_from_name(stem)

        output_dir = os.path.join(BASE, ds_folder, "analysis", stem)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n  ── {csv_file}  [{atk_type}]")

        try:
            df = load_csv(file_path, has_header)
        except Exception as e:
            print(f"     ERROR loading: {e}")
            continue

        # Payload bytes → int (needed for entropy / correlation plots)
        try:
            convert_payload_to_int(df)
        except Exception:
            pass  # signal-format files lack b0-b7; skip silently

        basic_statistics(df)

        # ── text statistics file ──────────────────────────────────
        save_statistics(df, output_dir, csv_file, attack_type=atk_type)

        # ── plots ─────────────────────────────────────────────────
        for fn, label in [
            (plot_can_id_distribution, "CAN ID distribution"),
            (plot_message_rate,        "message rate"),
            (plot_class_distribution,  "class distribution"),
            (plot_canid_vs_time,       "CAN ID vs time"),
            (plot_canid_periodicity,   "CAN ID periodicity"),
            (plot_byte_correlation,    "byte correlation"),
            (plot_payload_entropy,     "payload entropy"),
            (plot_label_vs_time,         "label vs time"),
            (plot_label_vs_time_windows, "label vs time windows"),
        ]:
            try:
                fn(df, output_dir)
            except Exception as e:
                print(f"     [warn] {label}: {e}")

        print(f"     Output → {output_dir}")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for ds_folder, modified_sub, has_header in DATASETS:
        run_dataset(ds_folder, modified_sub, has_header)

    print("\n\nDone. All statistics and plots have been saved.")
