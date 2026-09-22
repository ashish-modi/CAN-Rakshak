
from __future__ import annotations

import gc
import glob
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf  
        
from pathlib import Path as _P

from ids.base import IDS


def _evaluation(
    real_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[int, int, int, int, float | None, float, float | None]:
    """
    Compute binary classification metrics.

    Returns: (tn, fp, fn, tp, tpr, tnr, f1)
      - tpr and f1 are None when there are no positive ground-truth labels
        (e.g. the normal-traffic test split), to distinguish "undefined"
        from "zero".
      - A tiny epsilon (1e-9) avoids division-by-zero without affecting
        meaningful results.
    """
    real_labels      = np.asarray(real_labels)
    predicted_labels = np.asarray(predicted_labels)
    tn = int(((real_labels == 0) & (predicted_labels == 0)).sum())
    fp = int(((real_labels == 0) & (predicted_labels == 1)).sum())
    fn = int(((real_labels == 1) & (predicted_labels == 0)).sum())
    tp = int(((real_labels == 1) & (predicted_labels == 1)).sum())
    tnr = tn / (tn + fp + 1e-9)
    if tp + fn == 0:
        
        tpr, f1 = None, None
    else:
        tpr = tp / (tp + fn + 1e-9)
        f1  = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    return tn, fp, fn, tp, tpr, tnr, f1


_SYNCAN_ATTACKS = ["flooding", "plateau", "continuous", "playback", "suppress"]


_TIL_VALID_QUANTILE = 0.0001   

_TIL_XCANIDS_VALID_QUANTILE = 0.001  



class TIL(IDS):

    def __init__(self):
        self.threshold = None  


    def load(self, path: str):
        print("  [TIL] No weights to load (params in features/til/). Ready.")

    def save(self, path: str):
        pass

    def predict(self, X_test=None, **kwargs) -> np.ndarray:

        if self.threshold is None:
            raise RuntimeError("TIL.predict(): threshold not set. Call test() first.")
        if X_test is None or "z_itv" not in X_test.columns:
            raise ValueError("TIL.predict(): X_test must be a DataFrame with a 'z_itv' column.")
        return (X_test["z_itv"] < self.threshold).astype(int).to_numpy()


    def train(self, train_dataset_dir=None, X_train=None, Y_train=None, cfg=None, **kwargs):
  
        cfg = cfg or {}
        dataset_name = cfg.get("dataset_name", "SynCAN")
        project_root = os.path.normpath(os.path.join(cfg["dir_path"], ".."))
        til_dir      = os.path.join(project_root, "datasets", dataset_name, "features", "til")

        if dataset_name == "SynCAN":
            valid_path = os.path.join(til_dir, "syncan_til_train_valid.parquet")
            if not os.path.exists(valid_path):
                print("  [TIL] Validation parquet not found — run feature extraction first.")
                return
            df_valid       = pd.read_parquet(valid_path)
            self.threshold = float(np.quantile(df_valid["z_itv"], _TIL_VALID_QUANTILE))
            print(f"  [TIL] Threshold set from validation: {self.threshold:.6f}")
        else:
            raise NotImplementedError(f"TIL.train() not yet implemented for {dataset_name}")

    

    def _test_syncan(self, cfg: dict) -> dict:
        """
        SynCAN-specific test logic.
        Reads pre-computed z_itv parquets from features/til/,
        copies them to results/SynCAN/ with the naming eval_syncan_fusion.py expects,
        sets threshold, and computes per-attack TIL metrics.
        """
        project_root = os.path.normpath(os.path.join(cfg["dir_path"], ".."))
        til_dir      = os.path.join(project_root, "datasets", "SynCAN", "features", "til")
        results_dir  = os.path.join(project_root, "results", "SynCAN")
        os.makedirs(results_dir, exist_ok=True)

        
        valid_feat = os.path.join(til_dir, "syncan_til_train_valid.parquet")
        if not os.path.exists(valid_feat):
            raise FileNotFoundError(
                f"TIL: {valid_feat} not found. Run feature extraction first."
            )
        df_valid = pd.read_parquet(valid_feat)

        
        self.threshold = float(np.quantile(df_valid["z_itv"], _TIL_VALID_QUANTILE))
        print(f"  [TIL] Threshold = {self.threshold:.6f}  (q={_TIL_VALID_QUANTILE})")

        
        valid_out = os.path.join(results_dir, "Syncan_TIL_valid.parquet")
        df_valid.to_parquet(valid_out, index=False)
        print(f"  [TIL] Valid → {len(df_valid):,} rows saved to results/")

        
        eval_results: dict[str, dict] = {}
        for attack in _SYNCAN_ATTACKS:
            feat_path = os.path.join(til_dir, f"syncan_til_test_{attack}.parquet")
            if not os.path.exists(feat_path):
                print(f"  [TIL] [WARN] {os.path.basename(feat_path)} not found — skipping")
                continue

            df_attack = pd.read_parquet(feat_path)

            
            out_path = os.path.join(results_dir, f"Syncan_TIL_{attack}.parquet")
            df_attack.to_parquet(out_path, index=False)

            
            binary    = (df_attack["z_itv"] < self.threshold).astype(int).to_numpy()
            s_arr     = df_attack["Session"].to_numpy()

            tn, fp, fn, tp, tpr, tnr, f1 = _evaluation(s_arr, binary)
            eval_results[attack] = {
                "rows": int(len(df_attack)),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "TPR": None if tpr is None else float(tpr),
                "TNR": float(tnr),
                "F1":  None if f1  is None else float(f1),
            }
            print(f"  [TIL] {attack:12s} TPR={tpr or 0:.3f}  TNR={tnr:.3f}  F1={f1 or 0:.3f}")

        
        normal_feat = os.path.join(til_dir, "syncan_til_test_normal.parquet")
        if os.path.exists(normal_feat):
            df_normal = pd.read_parquet(normal_feat)
            df_normal.to_parquet(os.path.join(results_dir, "Syncan_TIL_normal.parquet"), index=False)
            binary = (df_normal["z_itv"] < self.threshold).astype(int).to_numpy()
            s_arr  = df_normal["Session"].to_numpy()
            tn, fp, fn, tp, tpr, tnr, f1 = _evaluation(s_arr, binary)
            eval_results["normal"] = {
                "rows": int(len(df_normal)),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "TPR": None if tpr is None else float(tpr),
                "TNR": float(tnr),
                "F1":  None if f1  is None else float(f1),
            }

        
        tprs = [v["TPR"] for v in eval_results.values() if v["TPR"] is not None]
        tnrs = [v["TNR"] for v in eval_results.values() if v["TNR"] is not None]
        f1s  = [v["F1"]  for v in eval_results.values() if v["F1"]  is not None]
        macro = {
            "TPR": float(np.mean(tprs)) if tprs else None,
            "TNR": float(np.mean(tnrs)) if tnrs else None,
            "F1":  float(np.mean(f1s))  if f1s  else None,
        }

        output = {
            "dataset":   "SynCAN",
            "model":     "TIL",
            "threshold": self.threshold,
            "by_attack": eval_results,
            "macro":     macro,
        }
        summary_path = os.path.join(results_dir, "Syncan_TIL_run_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"  [TIL] Summary → {summary_path}")
        return output

    

    def _test_xcanids(self, cfg: dict) -> dict:
        """
        X-CANIDS-specific TIL evaluation.
        Reads pre-computed z_itv parquets from features/til/,
        copies to results/X-CANIDS/ with the naming eval_xcanids_fusion.py expects,
        sets threshold from validation, computes per-variant metrics.
        """
        project_root = os.path.normpath(os.path.join(cfg["dir_path"], ".."))
        til_dir     = os.path.join(project_root, "datasets", "X-CANIDS", "features", "til")
        results_dir = os.path.join(project_root, "results", "X-CANIDS")
        os.makedirs(results_dir, exist_ok=True)

        valid_feat = None
        for candidate in ("xcanids_til_dump5.parquet",
                          "xcanids_til_sig107_dump5.parquet"):
            p = os.path.join(til_dir, candidate)
            if os.path.exists(p):
                valid_feat = p
                break
        if valid_feat is None:
            raise FileNotFoundError(
                f"TIL X-CANIDS: no valid TIL parquet found in {til_dir}. "
                "Run feature extraction first."
            )
        df_valid = pd.read_parquet(valid_feat)
        self.threshold = float(np.quantile(df_valid["z_itv"], _TIL_XCANIDS_VALID_QUANTILE))
        print(f"  [TIL] X-CANIDS Threshold = {self.threshold:.8f}  "
              f"(q={_TIL_XCANIDS_VALID_QUANTILE})  [{os.path.basename(valid_feat)}]")

        
        valid_out = os.path.join(results_dir, "X-CANIDS_TIL_valid.parquet")
        df_valid.to_parquet(valid_out, index=False)
        print(f"  [TIL] X-CANIDS valid → {len(df_valid):,} rows saved to results/")

        eval_results: dict[str, dict] = {}
        feat_parquets = sorted(
            list(Path(til_dir).glob("xcanids_til_dump6-*.parquet")) +
            list(Path(til_dir).glob("xcanids_til_sig107_dump6-*.parquet"))
        )

        for feat_path in feat_parquets:
            stem = feat_path.stem  
            
            idx = stem.find("dump6-")
            if idx == -1:
                print(f"  [TIL] [WARN] Cannot parse tag from {stem} — skipping")
                continue
            tag = stem[idx + len("dump6-"):]  

            df_test = pd.read_parquet(feat_path)

            out_path = os.path.join(results_dir, f"X-CANIDS_TIL_{tag}.parquet")
            df_test.to_parquet(out_path, index=False)

            
            s_arr  = (df_test["Session"].to_numpy() != 0).astype(int)
            binary = (df_test["z_itv"].to_numpy() < self.threshold).astype(int)

            tn, fp, fn, tp, tpr, tnr, f1 = _evaluation(s_arr, binary)
            eval_results[tag] = {
                "rows": int(len(df_test)),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "TPR": None if tpr is None else float(tpr),
                "TNR": float(tnr),
                "F1":  None if f1  is None else float(f1),
            }
            print(f"  [TIL] {tag:30s}  TPR={tpr or 0:.3f}  TNR={tnr:.3f}  F1={f1 or 0:.3f}")

        tprs = [v["TPR"] for v in eval_results.values() if v["TPR"] is not None]
        tnrs = [v["TNR"] for v in eval_results.values() if v["TNR"] is not None]
        f1s  = [v["F1"]  for v in eval_results.values() if v["F1"]  is not None]
        macro = {
            "TPR": float(np.mean(tprs)) if tprs else None,
            "TNR": float(np.mean(tnrs)) if tnrs else None,
            "F1":  float(np.mean(f1s))  if f1s  else None,
        }

        output = {
            "dataset":   "X-CANIDS",
            "model":     "TIL",
            "threshold": self.threshold,
            "by_variant": eval_results,
            "macro":      macro,
        }
        summary_path = os.path.join(results_dir, "X-CANIDS_TIL_run_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"  [TIL] X-CANIDS summary → {summary_path}")
        return output


    def _test_road(self, cfg: dict) -> dict:
      
        project_root = os.path.normpath(os.path.join(cfg["dir_path"], ".."))
        features_dir = _P(project_root) / "datasets" / "ROAD" / "features"
        til_dir      = features_dir / "til"
        results_dir  = _P(project_root) / "results" / "ROAD"
        results_dir.mkdir(parents=True, exist_ok=True)

        _TIL_ROAD_QUANTILE  = 0.001   
        _MIN_NORMAL_TIL     = 200     

        
        _ATTACK_ID_MAP: dict[str, str] = {
            "correlated_signal":  "1760",
            "max_speedometer":    "208",
            "reverse_light_off":  "208",
            "reverse_light_on":   "208",
            "max_engine_coolant": "1255",
        }
        _FOCUSED_COL: dict[str, str] = {
            "1760": "z_itv_1760",
            "208":  "z_itv_208",
            "1255": "z_itv_1255",
        }
        _NO_LABEL_KEYS = {"accelerator", "fuzzing"}

        def _attack_family(stem: str) -> str:
            """Return the attack-family keyword from a file stem."""
            for kw in _ATTACK_ID_MAP:
                if kw in stem:
                    return kw
            return ""

        
        valid_til_files = sorted(til_dir.glob("road_til_ambient_*.parquet"))
        if not valid_til_files:
            raise FileNotFoundError(
                f"TIL ROAD: no road_til_ambient_*.parquet found in {til_dir}.\n"
                "Run CanivalExtractor (Stage 1) first."
            )

        
        valid_pool: dict[str, list[np.ndarray]] = {}   

        for vf in valid_til_files:
            original_stem = vf.stem.removeprefix("road_til_")
            out_name = f"ROAD_TIL_focused_valid_{original_stem}.parquet"
            df_v = pd.read_parquet(vf)
            df_v.to_parquet(results_dir / out_name, index=False)
            print(f"  [TIL] ROAD valid: {vf.name} → {out_name}  ({len(df_v):,} rows)")
            normal_rows = df_v.loc[df_v["Session"] == 0]
            for col in ["z_itv", "z_itv_208", "z_itv_1255", "z_itv_1760"]:
                if col in normal_rows.columns:
                    arr = normal_rows[col].dropna().to_numpy()
                    if len(arr):
                        valid_pool.setdefault(col, []).append(arr)

        
        valid_pool_cat: dict[str, np.ndarray] = {
            col: np.concatenate(arrs) for col, arrs in valid_pool.items()
        }

        
        if "z_itv" in valid_pool_cat:
            self.threshold = float(
                np.quantile(valid_pool_cat["z_itv"], _TIL_ROAD_QUANTILE)
            )
        else:
            self.threshold = 0.0
        print(
            f"  [TIL] ROAD global z_itv threshold (p{_TIL_ROAD_QUANTILE*100:.1f}) "
            f"= {self.threshold:.8f}  "
            f"({len(valid_pool_cat.get('z_itv', [])):,} normal rows)"
        )

        def _adaptive_threshold(df: pd.DataFrame, col: str) -> tuple[float, bool]:
            """
            Compute 0.1th-percentile threshold from normal rows of this file.
            Falls back to valid-pool if < _MIN_NORMAL_TIL normal rows present.
            Returns (threshold, fallback_used).
            """
            normal_vals = df.loc[df["Session"] == 0, col].dropna().to_numpy()
            if len(normal_vals) >= _MIN_NORMAL_TIL:
                return float(np.quantile(normal_vals, _TIL_ROAD_QUANTILE)), False
            
            pool = valid_pool_cat.get(col, np.array([]))
            if len(pool):
                return float(np.quantile(pool, _TIL_ROAD_QUANTILE)), True
            
            return self.threshold, True

        
        attack_til_files = sorted(
            p for p in til_dir.glob("road_til_*.parquet")
            if not p.name.startswith("road_til_ambient_")
        )

        records: list[dict] = []
        for tf_file in attack_til_files:
            original_stem = tf_file.stem.removeprefix("road_til_")
            out_name = f"ROAD_TIL_focused_{original_stem}.parquet"
            df_t = pd.read_parquet(tf_file)
            df_t.to_parquet(results_dir / out_name, index=False)

            
            family      = _attack_family(original_stem)
            attacked_id = _ATTACK_ID_MAP.get(family)
            focused_col = _FOCUSED_COL.get(attacked_id, "") if attacked_id else ""

            use_focused = (
                focused_col
                and focused_col in df_t.columns
                and not df_t[focused_col].isna().all()
            )
            det_col  = focused_col if use_focused else "z_itv"
            col_note = f"z_itv_ID{attacked_id}" if use_focused else "z_itv_global"

            if not use_focused and family and family not in _NO_LABEL_KEYS:
                print(f"  [WARN] {original_stem}: focused col {focused_col} missing, "
                      f"falling back to global z_itv")

            
            thr, fb_used = _adaptive_threshold(df_t, det_col)

            
            
            no_label = any(k in original_stem for k in _NO_LABEL_KEYS)

            s_arr  = df_t["Session"].to_numpy().astype(int)
            z_arr  = df_t[det_col].fillna(1.0).to_numpy()
            
            binary = (z_arr < thr).astype(int)

            
            tn, fp_cnt, fn, tp, tpr, tnr, f1 = _evaluation(s_arr, binary)

            
            _W, _K = 200, 1
            n      = len(s_arr)
            n_win  = n // _W
            w_tpr = w_tnr = w_f1 = None
            if n_win > 0:
                lbl_mat = s_arr[:n_win * _W].reshape(n_win, _W)
                prd_mat = binary[:n_win * _W].reshape(n_win, _W)
                win_lbl = lbl_mat.max(axis=1)
                win_prd = (prd_mat.sum(axis=1) >= _K).astype(int)
                _, _, _, _, w_tpr, w_tnr, w_f1 = _evaluation(win_lbl, win_prd)

            rec = {
                "file":        tf_file.name,
                "out_name":    out_name,
                "rows":        int(len(df_t)),
                "det_col":     det_col,
                "threshold":   round(thr, 8),
                "fallback":    fb_used,
                "tn": tn, "fp": fp_cnt, "fn": fn, "tp": tp,
                "TPR":     None if (tpr   is None or no_label) else float(tpr),
                "TNR":     float(tnr),
                "F1":      None if (f1    is None or no_label) else float(f1),
                "W_TPR":   None if (w_tpr is None or no_label) else float(w_tpr),
                "W_TNR":   None if w_tnr  is None else float(w_tnr),
                "W_F1":    None if (w_f1  is None or no_label) else float(w_f1),
            }
            records.append(rec)
            fb_str   = " [fb]" if fb_used else ""
            if no_label:
                print(
                    f"  [TIL] {original_stem:55s}  "
                    f"TPR=N/A  TNR={tnr:.3f}  W-TPR=N/A  W-TNR={w_tnr or 0:.3f}"
                    f"  [{col_note} thr={thr:.4f}{fb_str}]"
                )
            else:
                tpr_s  = f"{tpr  or 0:.3f}"
                f1_s   = f"{f1   or 0:.3f}"
                wtpr_s = f"{w_tpr or 0:.3f}" if w_tpr is not None else "N/A"
                wf1_s  = f"{w_f1  or 0:.3f}" if w_f1  is not None else "N/A"
                print(
                    f"  [TIL] {original_stem:55s}  "
                    f"TPR={tpr_s}  TNR={tnr:.3f}  F1={f1_s} │ "
                    f"W-TPR={wtpr_s}  W-TNR={w_tnr or 0:.3f}  W-F1={wf1_s}"
                    f"  [{col_note} thr={thr:.4f}{fb_str}]"
                )

        output = {
            "dataset":           "ROAD",
            "model":             "TIL",
            "global_threshold":  self.threshold,
            "threshold_quantile": _TIL_ROAD_QUANTILE,
            "by_file":           {r["file"]: r for r in records},
        }
        summary_path = results_dir / "ROAD_TIL_run_summary.json"
        with open(str(summary_path), "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"  [TIL] ROAD summary → {summary_path}")
        return output

    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs) -> dict:
        """
        Read pre-computed TIL z_itv scores, set threshold from validation,
        save result parquets to results/{dataset_name}/, compute per-attack metrics.
        """
        cfg = cfg or {}
        dataset_name = cfg.get("dataset_name", "SynCAN")

        if dataset_name == "SynCAN":
            return self._test_syncan(cfg)
        elif dataset_name == "X-CANIDS":
            return self._test_xcanids(cfg)
        elif dataset_name == "ROAD":
            return self._test_road(cfg)
        else:
            raise NotImplementedError(
                f"TIL.test() not yet implemented for dataset: {dataset_name}"
            )







_XCANIDS_WINDOW_SIZE = 1        
_XCANIDS_EPOCH       = 10       
_XCANIDS_N_SLICES    = 15       
_XCANIDS_BATCH_SIZE  = 250
_XCANIDS_TIME_CUTOFF = _XCANIDS_WINDOW_SIZE + 1  

_XCANIDS_ID_MPS: dict[str, int] = {
    "1151": 50,  "1265": 50,  "128": 100,  "129": 100,
    "1292": 10,  "1322": 10,  "1345": 11,  "1349": 10,
    "1351": 10,  "1353": 10,  "1363": 5,   "1365": 10,
    "1366": 10,  "1367": 10,  "1419": 10,  "1427": 5,
    "1440": 1,   "1456": 1,   "273": 100,  "274": 100,
    "275": 100,  "354": 100,  "399": 100,  "512": 100,
    "544": 100,  "593": 100,  "608": 100,  "68": 1,
    "688": 100,  "790": 100,  "809": 100,  "897": 50,
    "899": 50,   "902": 50,   "903": 50,
}
_XCANIDS_ID_NSIG: OrderedDict[str, int] = OrderedDict([
    ("1151", 1), ("1265", 3), ("128",  5), ("129",  3),
    ("1292", 2), ("1322", 1), ("1345", 3), ("1349", 4),
    ("1351", 2), ("1353", 4), ("1363", 4), ("1365", 1),
    ("1366", 5), ("1367", 2), ("1419", 5), ("1427", 4),
    ("1440", 1), ("1456", 1), ("273",  4), ("274",  3),
    ("275",  4), ("354",  3), ("399",  3), ("512",  1),
    ("544",  4), ("593",  2), ("608",  5), ("68",   2),
    ("688",  2), ("790",  6), ("809",  6), ("897",  1),
    ("899",  2), ("902",  4), ("903",  4),
])
_XCANIDS_FIXED_IDS: list[str] = sorted(_XCANIDS_ID_NSIG.keys())
_XCANIDS_N_SIGS:    int       = sum(_XCANIDS_ID_NSIG.values())


_XCANIDS_CANET_VALID_QUANTILE = 0.9999   




_SYNCAN_ID_MPS: dict[str, int] = {
    "id1": 67, "id10": 22, "id2": 33, "id3": 67,
    "id4": 22, "id5": 67, "id6": 33, "id7": 67,
    "id8": 67, "id9": 33,
}
_SYNCAN_ID_NSIG: OrderedDict[str, int] = OrderedDict([
    ("id1",  2), ("id10", 4), ("id2",  3), ("id3",  2),
    ("id4",  1), ("id5",  2), ("id6",  2), ("id7",  2),
    ("id8",  1), ("id9",  1),
])
_SYNCAN_FIXED_IDS: list[str] = sorted(_SYNCAN_ID_NSIG.keys())
_SYNCAN_N_SIGS:    int       = sum(_SYNCAN_ID_NSIG.values())



_CANET_VALID_QUANTILE = 0.999   









def _build_syncan_canet(h: int = 5):
    """Build the SynCAN CANet architecture. Returns a keras.Model."""
    from tensorflow import keras
    fixed_ids = _SYNCAN_FIXED_IDS
    id_nsig   = _SYNCAN_ID_NSIG
    id_mps    = _SYNCAN_ID_MPS
    n_sigs    = _SYNCAN_N_SIGS

    inputs = {
        cid: keras.Input(shape=(id_mps[cid], id_nsig[cid]), name=cid)
        for cid in fixed_ids
    }
    lstms = [
        keras.layers.LSTM(h * id_nsig[cid], name=f"lstm_{cid}")
        for cid in fixed_ids
    ]
    x_id = [lstms[i](inputs[cid]) for i, cid in enumerate(fixed_ids)]
    x    = keras.layers.Concatenate()(x_id)
    x    = keras.layers.Dense((h * n_sigs) // 2, activation="elu")(x)
    x    = keras.layers.Dense(n_sigs - 1,         activation="elu")(x)
    out  = keras.layers.Dense(n_sigs,              activation="elu")(x)
    return keras.Model(inputs, out)




def _load_syncan_npz(
    npz_path: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load a syncan_canet_{stem}.npz file produced by CanivalExtractor."""
    data       = np.load(npz_path, allow_pickle=False)
    x_dict     = {cid: data[cid] for cid in _SYNCAN_FIXED_IDS}
    y          = data["y"]
    time_label = data["time_and_labels"]
    return x_dict, y, time_label




def _build_xcanids_canet(h: int = 5):
    """Build the X-CANIDS CANet architecture. Returns a keras.Model."""
    from tensorflow import keras
    fixed_ids = _XCANIDS_FIXED_IDS
    id_nsig   = _XCANIDS_ID_NSIG
    id_mps    = _XCANIDS_ID_MPS
    n_sigs    = _XCANIDS_N_SIGS

    inputs = {
        cid: keras.Input(shape=(id_mps[cid], id_nsig[cid]), name=cid)
        for cid in fixed_ids
    }
    lstms = [
        keras.layers.LSTM(h * id_nsig[cid], name=f"lstm_{cid}")
        for cid in fixed_ids
    ]
    x_id = [lstms[i](inputs[cid]) for i, cid in enumerate(fixed_ids)]
    x    = keras.layers.Concatenate()(x_id)
    x    = keras.layers.Dense((h * n_sigs) // 2, activation="elu")(x)
    x    = keras.layers.Dense(n_sigs - 1,         activation="elu")(x)
    out  = keras.layers.Dense(n_sigs,              activation="elu")(x)
    return keras.Model(inputs, out)




def _xcanids_load_arrange(file_path: Path) -> pd.DataFrame:
    """
    Load a sig107 parquet and cast ID to str.
    Mirrors load_arrange_data from run_canet_xcanids.py.
    """
    df = pd.read_parquet(file_path)
    df.reset_index(drop=True, inplace=True)
    df["ID"] = df["ID"].astype(str)
    return df


def _xcanids_get_repeated_sequences(
    data: pd.DataFrame, can_id: str, n_sig: int, n_step: int
) -> np.ndarray:
    """
    Build sliding-window sequences for a single CAN ID.
    Mirrors get_repeated_sequences from run_canet_xcanids.py exactly.
    """
    sig_columns = [f"Signal{i}" for i in range(1, n_sig + 1)]
    df_id  = data.loc[data["ID"] == can_id, ["Idx", "Session"] + sig_columns]
    np_sig = df_id[["Session"] + sig_columns].to_numpy()
    np_seq = np.lib.stride_tricks.sliding_window_view(np_sig, window_shape=n_step, axis=0)
    np_seq = np_seq.swapaxes(1, 2)
    np_seq = np_seq[:, :, 1:]          
    n_seq  = np_seq.shape[0]
    end_idx    = data["Idx"].iloc[-1]
    n_repeats  = np.diff(df_id["Idx"].to_list() + [end_idx])[-n_seq:]
    return np.repeat(np_seq, n_repeats, axis=0)


def _xcanids_prepare_dataset(
    file_path: Path, time_cutoff: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Build per-CAN-ID sliding-window arrays and time/label vector.
    Mirrors prepare_dataset from run_canet_xcanids.py exactly.
    """
    data       = _xcanids_load_arrange(file_path)
    data       = data.reset_index(names="Idx")
    time_start = data["Time"].iloc[0]
    n_rows_to_use = data.loc[data["Time"] > time_start + time_cutoff, "Time"].shape[0]
    time_and_labels = data.loc[
        data["Time"] > time_start + time_cutoff, ["Time", "Session"]
    ].to_numpy()

    data_dict: dict[str, np.ndarray] = {}
    for can_id, nsig in _XCANIDS_ID_NSIG.items():
        seq = _xcanids_get_repeated_sequences(
            data, can_id, nsig, _XCANIDS_ID_MPS[can_id]
        )
        data_dict[can_id] = seq[-n_rows_to_use:].copy()

    common_rows = min([n_rows_to_use] + [s.shape[0] for s in data_dict.values()]) \
                  if data_dict else n_rows_to_use
    if common_rows <= 0:
        raise ValueError(f"No aligned CANet rows for {file_path}")
    for cid in list(data_dict.keys()):
        data_dict[cid] = data_dict[cid][-common_rows:].copy()
    time_and_labels = time_and_labels[-common_rows:].copy()
    return data_dict, time_and_labels


def _xcanids_load_inputs(
    data_path: Path, time_cutoff: int
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Prepare sliding-window inputs and extract y (last-timestep signals)."""
    x_dict, x_time_label = _xcanids_prepare_dataset(data_path, time_cutoff)
    y = np.concatenate(
        [x_dict[cid][:, -1, :] for cid in _XCANIDS_FIXED_IDS], axis=1
    )
    return x_dict, y, x_time_label


def _xcanids_slice_data(
    file_path: Path, n_sliced: int, cache_dir: Path
) -> list[Path]:
    """
    Slice a large sig107 parquet into n_sliced pieces for memory efficiency.
    Special-cases Suspension attacks with fake-message insertion.
    Mirrors slice_data_for_canet from run_canet_xcanids.py exactly.
    """
    

    try:
        attack = file_path.stem.split("-")[1]
    except IndexError:
        attack = "normal"

    df = pd.read_parquet(file_path)
    p  = len(df) // n_sliced
    split_indices = list(range(0, len(df) + 1, p))[:-1]

    paths: list[Path] = []
    latest_msg = None
    assertion  = False
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_sliced):
        if i + 1 < n_sliced:
            sliced_df = df.iloc[split_indices[i]: split_indices[i + 1]]
        else:
            sliced_df = df.iloc[split_indices[i]:]

        if attack == "susp":
            
            
            
            
            
            
            t_start = sliced_df["Time"].min()
            t_end   = sliced_df["Time"].max()
            try:
                
                
                target_id = int(file_path.stem.split("-")[2][:-1], 16)
                msgs = sliced_df.loc[sliced_df["ID"] == target_id]

                if len(msgs) == 0 or 480 < t_start < 1440:
                    
                    
                    assert latest_msg is not None
                    t_end  = min(t_end, 1440)      
                    
                    n_fake = _XCANIDS_ID_MPS[str(target_id)] * int(round(t_end - t_start, 0))
                    if n_fake > 0:
                        
                        df_fake = pd.concat([latest_msg] * n_fake, axis=0)
                        df_fake["Time"]    = np.linspace(t_start, t_end, num=n_fake)
                        df_fake["Session"] = 1   
                        sliced_df = pd.concat([sliced_df, df_fake])
                        
                        dups = sliced_df.duplicated(subset=["Time"], keep="first")
                        if dups.sum() > 0:
                            sliced_df = sliced_df[~dups]
                        sliced_df.sort_values("Time", ignore_index=True, inplace=True)
                else:
                    
                    latest_msg = msgs.tail(1)
            except (AssertionError, ValueError, KeyError):
                assertion = True

        save_path = cache_dir / f"{file_path.stem}_{i + 1}.parquet"
        sliced_df.to_parquet(save_path)
        paths.append(save_path)

    if assertion:
        print(f"  [CANet] [INFO] Suspension target ID not found — fake insertion skipped for {file_path.name}")
    return paths




class CANet(IDS):
    """
    Multi-input LSTM autoencoder for CAN bus anomaly detection.
    One LSTM branch per CAN ID; anomaly score = per-message MSE.

    Supports SynCAN, X-CANIDS, and ROAD datasets via dataset-specific
    _test_syncan() / _test_xcanids() / _test_road() methods.
    Dispatched by test(cfg={"dataset_name": ...}).
    """

    def __init__(self):
        self.model         = None
        self.threshold     = None
        self._weights_path = None   

    

    def load(self, path: str):
        """
        Store the weights path for deferred loading.
        Actual model building and weight loading happen in test() where
        dataset-specific constants from cfg are available.
        """
        self._weights_path = path
        print(f"  [CANet] Weights path stored: {path}")

    def save(self, path: str):
        """Save current model weights (TF checkpoint format)."""
        if self.model is None:
            raise RuntimeError("CANet.save(): no model to save.")
        
        clean = path[:-3] if path.endswith(".h5") else path
        self.model.save_weights(clean)
        print(f"  [CANet] Weights saved → {clean}")

    

    def predict(self, X_test: dict[str, np.ndarray] | None = None, batch_size: int = 250, **kwargs) -> np.ndarray:
        """
        Run model on x_dict (dict of per-ID window arrays).
        Returns MSE array of shape (n_rows,).
        """
        if self.model is None:
            raise RuntimeError("CANet.predict(): model not loaded. Call test() or build the model first.")
        import tensorflow as tf
        ds      = tf.data.Dataset.from_tensor_slices(X_test)
        ds      = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        y_pred  = self.model.predict(ds, verbose=0)
        return y_pred

    

    def train(self, train_dataset_dir=None, X_train=None, Y_train=None, cfg=None, **kwargs):
        """
        Train CANet from scratch on SynCAN NPZ features.
        (Training is optional for SynCAN — pre-trained weights are provided.)
        """
        import tensorflow as tf
        cfg = cfg or {}
        dataset_name = cfg.get("dataset_name", "SynCAN")
        if dataset_name != "SynCAN":
            raise NotImplementedError(f"CANet.train() not yet implemented for {dataset_name}")

        project_root = os.path.normpath(os.path.join(cfg["dir_path"], ".."))
        canet_dir    = os.path.join(project_root, "datasets", dataset_name, "features", "canet")
        epochs       = cfg.get("epochs", 5)
        batch_size   = cfg.get("batch_size", 250)

        
        self.model = _build_syncan_canet(h=5)
        mse_fn     = tf.keras.losses.MeanSquaredError()
        self.model.compile(optimizer="adam", loss=mse_fn)
        print(f"  [CANet] Training for {epochs} epoch(s) on SynCAN …")

        for ep in range(1, epochs + 1):
            train_stems = ["train_1", "train_2", "train_3", "train_4"]
            for stem in train_stems:
                npz = os.path.join(canet_dir, f"syncan_canet_{stem}.npz")
                if not os.path.exists(npz):
                    print(f"    [WARN] {npz} not found — skipping")
                    continue
                x_dict, y, _ = _load_syncan_npz(npz)
                ds = tf.data.Dataset.from_tensor_slices((x_dict, y))
                ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                self.model.fit(ds, epochs=1, verbose=1)
                del ds
                gc.collect()
            print(f"  [CANet] Epoch {ep}/{epochs} complete")

        print("  [CANet] Training done.")

    

    def _test_syncan(self, cfg: dict) -> dict:
        """SynCAN-specific test logic. Mirrors run_canet_syncan.py + eval_syncan_fusion.py."""
        import tensorflow as tf
        from tensorflow import keras

        project_root = os.path.normpath(os.path.join(cfg["dir_path"], ".."))
        canet_dir    = os.path.join(project_root, "datasets", "SynCAN", "features", "canet")
        results_dir  = os.path.join(project_root, "results", "SynCAN")
        os.makedirs(results_dir, exist_ok=True)

        
        self.model = _build_syncan_canet(h=5)

        
        
        models_subdir = os.path.join(project_root, "models", "SynCAN")
        index_files   = sorted(glob.glob(os.path.join(models_subdir, "*.index")))
        if index_files:
            weights_ckpt = index_files[0].replace(".index", "")
        elif self._weights_path:
            weights_ckpt = (
                self._weights_path[:-3]
                if self._weights_path.endswith(".h5")
                else self._weights_path
            )
        else:
            raise FileNotFoundError(
                "CANet SynCAN: no weights found in models/SynCAN/. "
                "Copy canet_syncan_epoch05.{index,data-*} there first."
            )
        self.model.load_weights(weights_ckpt)
        print(f"  [CANet] Loaded weights: {os.path.basename(weights_ckpt)}")

        mse_fn = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )

        def _run_file(npz_path: str, batch_size: int = 250) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Load one NPZ, run CANet inference, return (mse_vals, timestamps, labels).
            tf.device("CPU") avoids GPU-memory fragmentation when looping over many
            files.  The dataset and predictions are deleted immediately after MSE
            computation to keep RAM usage flat across all test files.
            """
            x_dict, y_true, time_label = _load_syncan_npz(npz_path)
            with tf.device("CPU"):
                ds     = tf.data.Dataset.from_tensor_slices(x_dict)
                ds     = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                y_pred = self.model.predict(ds, verbose=0)
            mse_vals = mse_fn(y_true, y_pred).numpy()
            del ds, y_pred
            gc.collect()
            return mse_vals, time_label[:, 0], time_label[:, 1].astype(int)

        
        valid_npz  = os.path.join(canet_dir, "syncan_canet_train_valid.npz")
        mse_v, t_v, s_v = _run_file(valid_npz)
        self.threshold   = float(np.quantile(mse_v, _CANET_VALID_QUANTILE))

        valid_result = pd.DataFrame({"Time": t_v, "MSE": mse_v, "Session": s_v})
        valid_result["Time"]    = valid_result["Time"].round(7)
        valid_result["Session"] = valid_result["Session"].astype(int)
        valid_path = os.path.join(results_dir, "Syncan_CANet_valid.parquet")
        valid_result.to_parquet(valid_path, index=False)
        print(f"  [CANet] Valid → {len(valid_result):,} rows | threshold={self.threshold:.6f}")

        
        eval_results: dict[str, dict] = {}
        for attack in _SYNCAN_ATTACKS:
            npz_path = os.path.join(canet_dir, f"syncan_canet_test_{attack}.npz")
            if not os.path.exists(npz_path):
                print(f"  [CANet] [WARN] {os.path.basename(npz_path)} not found — skipping")
                continue

            mse_vals, t_arr, s_arr = _run_file(npz_path)
            binary = (mse_vals >= self.threshold).astype(int)

            result_df = pd.DataFrame({"Time": t_arr, "MSE": mse_vals, "Session": s_arr})
            result_df["Time"]    = result_df["Time"].round(7)
            result_df["Session"] = result_df["Session"].astype(int)
            out_path = os.path.join(results_dir, f"Syncan_CANet_{attack}.parquet")
            result_df.to_parquet(out_path, index=False)

            tn, fp, fn, tp, tpr, tnr, f1 = _evaluation(s_arr, binary)
            eval_results[attack] = {
                "rows": int(len(result_df)),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "TPR": None if tpr is None else float(tpr),
                "TNR": float(tnr),
                "F1":  None if f1  is None else float(f1),
            }
            print(f"  [CANet] {attack:12s} TPR={tpr or 0:.3f}  TNR={tnr:.3f}  F1={f1 or 0:.3f}")

        
        normal_npz = os.path.join(canet_dir, "syncan_canet_test_normal.npz")
        if os.path.exists(normal_npz):
            mse_vals, t_arr, s_arr = _run_file(normal_npz)
            result_df = pd.DataFrame({"Time": t_arr, "MSE": mse_vals, "Session": s_arr})
            result_df["Time"]    = result_df["Time"].round(7)
            result_df["Session"] = result_df["Session"].astype(int)
            result_df.to_parquet(os.path.join(results_dir, "Syncan_CANet_normal.parquet"), index=False)
            binary = (mse_vals >= self.threshold).astype(int)
            tn, fp, fn, tp, tpr, tnr, f1 = _evaluation(s_arr, binary)
            eval_results["normal"] = {
                "rows": int(len(result_df)),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "TPR": None if tpr is None else float(tpr),
                "TNR": float(tnr),
                "F1":  None if f1  is None else float(f1),
            }

        
        tprs = [v["TPR"] for v in eval_results.values() if v["TPR"] is not None]
        tnrs = [v["TNR"] for v in eval_results.values() if v["TNR"] is not None]
        f1s  = [v["F1"]  for v in eval_results.values() if v["F1"]  is not None]
        macro = {
            "TPR": float(np.mean(tprs)) if tprs else None,
            "TNR": float(np.mean(tnrs)) if tnrs else None,
            "F1":  float(np.mean(f1s))  if f1s  else None,
        }

        output = {
            "dataset":   "SynCAN",
            "model":     "CANet",
            "threshold": self.threshold,
            "by_attack": eval_results,
            "macro":     macro,
        }
        summary_path = os.path.join(results_dir, "Syncan_CANet_run_summary.json")
        with open(summary_path, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"  [CANet] Summary → {summary_path}")
        return output

    

    def _test_xcanids(self, cfg: dict) -> dict:
        """
        X-CANIDS-specific test logic.
        Mirrors run_canet_xcanids.py (main + run_prediction + slice_data_for_canet).
        Reads sig107 parquets from datasets/X-CANIDS/modified_dataset/,
        slices them into 15 chunks, runs CANet inference, saves
        X-CANIDS_CANet_*.parquet to results/X-CANIDS/.
        """
        import tensorflow as tf

        project_root = os.path.normpath(os.path.join(cfg["dir_path"], ".."))
        mod_dir      = Path(project_root) / "datasets" / "X-CANIDS" / "modified_dataset"
        results_dir  = Path(project_root) / "results" / "X-CANIDS"
        cache_dir    = mod_dir / "cache"
        results_dir.mkdir(parents=True, exist_ok=True)

        
        self.model = _build_xcanids_canet(h=5)

        
        models_subdir = os.path.join(project_root, "models", "X-CANIDS")
        index_files   = sorted(glob.glob(os.path.join(models_subdir, "*.index")))
        if index_files:
            weights_ckpt = index_files[0].replace(".index", "")
        elif self._weights_path:
            weights_ckpt = (
                self._weights_path[:-3]
                if self._weights_path.endswith(".h5")
                else self._weights_path
            )
        else:
            raise FileNotFoundError(
                "CANet X-CANIDS: no weights found in models/X-CANIDS/. "
                "Copy canet_xcanids_epoch10.{index,data-*} there first."
            )
        self.model.load_weights(weights_ckpt)
        print(f"  [CANet] Loaded X-CANIDS weights: {os.path.basename(weights_ckpt)}")

        mse_fn = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )

        def _run_file(file_path: Path, tag: str) -> pd.DataFrame:
            """Slice file → inference on each slice → concatenate results."""
            sliced = _xcanids_slice_data(file_path, _XCANIDS_N_SLICES, cache_dir)
            frames: list[pd.DataFrame] = []
            for sliced_file in sliced:
                x_dict, y_true, tl = _xcanids_load_inputs(
                    sliced_file, _XCANIDS_TIME_CUTOFF
                )
                x_time  = tl[:, 0]
                x_label = tl[:, 1].astype(int)
                with tf.device("CPU"):
                    ds = tf.data.Dataset.from_tensor_slices(x_dict)
                    ds = ds.batch(_XCANIDS_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
                y_pred = self.model.predict(ds, verbose=0)
                mse_v  = mse_fn(y_true, y_pred).numpy()
                frames.append(
                    pd.DataFrame(
                        {"Time": x_time, "MSE": mse_v, "Session": x_label},
                        dtype=float,
                    ).assign(
                        Time=lambda d: d["Time"].round(6),
                        Session=lambda d: d["Session"].astype(int),
                    )
                )
                del ds
                gc.collect()
            return pd.concat(frames, axis=0, ignore_index=True)

        
        valid_file = mod_dir / "sig107_dump5.parquet"
        if not valid_file.exists():
            raise FileNotFoundError(
                f"CANet X-CANIDS: {valid_file} not found. Run feature extraction first."
            )
        valid_df = _run_file(valid_file, "valid")
        self.threshold = float(np.quantile(valid_df["MSE"].to_numpy(), _XCANIDS_CANET_VALID_QUANTILE))
        valid_out = results_dir / "X-CANIDS_CANet_valid.parquet"
        valid_df.to_parquet(valid_out, index=False)
        print(f"  [CANet] X-CANIDS valid → {len(valid_df):,} rows | threshold={self.threshold:.6e}")

        
        test_files = sorted(mod_dir.glob("sig107_dump6-*.parquet"))
        records: list[dict] = []
        for test_file in test_files:
            
            parts = test_file.stem.split("-")
            tag   = "-".join(parts[1:]) if len(parts) > 1 else "unknown"
            out_path = results_dir / f"X-CANIDS_CANet_{tag}.parquet"

            result_df = _run_file(test_file, tag)
            result_df.to_parquet(out_path, index=False)

            
            mse_arr = result_df["MSE"].to_numpy()
            s_arr   = result_df["Session"].to_numpy()
            binary  = (mse_arr >= self.threshold).astype(int)
            tn, fp, fn, tp, tpr, tnr, f1 = _evaluation(s_arr, binary)
            rec = {
                "tag": tag, "rows": int(len(result_df)),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "TPR": None if tpr is None else float(tpr),
                "TNR": float(tnr),
                "F1":  None if f1  is None else float(f1),
            }
            records.append(rec)
            print(f"  [CANet] {tag:30s}  TPR={tpr or 0:.3f}  F1={f1 or 0:.3f}")

        output = {
            "dataset":   "X-CANIDS",
            "model":     "CANet",
            "threshold": self.threshold,
            "by_variant": {r["tag"]: r for r in records},
        }
        summary_path = results_dir / "X-CANIDS_CANet_run_summary.json"
        with open(summary_path, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"  [CANet] X-CANIDS summary → {summary_path}")
        return output

    

    def _test_road(self, cfg: dict) -> dict:
        """
        ROAD-specific test logic.

        Mirrors run_canet_road.py + eval_road_fusion.py.

        Key differences from SynCAN / X-CANIDS:
          - IDs are DYNAMIC — loaded from features/road_config.json at runtime.
            (The ROAD dataset has 64 CAN IDs; hardcoding them would be brittle.)
          - Parquets are split into N_SLICES chunks via pyarrow iter_batches
            to avoid loading 64 × full-file np.repeat arrays into RAM.
          - Each row has both Session (time-window label) AND Label (exact
            injected-frame flag).  Both are saved in the output parquet.
          - Per-ID MSE columns (MSE_208, MSE_1255, MSE_1760) are saved
            alongside the global MSE for the three attack-focused CAN IDs.
          - Threshold is adaptive: 99.9th percentile of validation normal rows
            (Session == 0), recomputed fresh from the valid_* parquets.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        import tensorflow as tf

        
        project_root = os.path.normpath(os.path.join(cfg["dir_path"], ".."))
        features_dir = Path(project_root) / "datasets" / "ROAD" / "features"
        canet_dir    = features_dir / "canet"
        results_dir  = Path(project_root) / "results" / "ROAD"
        cache_dir    = canet_dir / "cache"
        results_dir.mkdir(parents=True, exist_ok=True)

        
        config_path = features_dir / "road_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"ROAD config not found: {config_path}.\n"
                "Run CanivalExtractor (Stage 1) first."
            )
        with open(config_path) as fh:
            road_cfg = json.load(fh)

        
        
        
        from collections import OrderedDict as _OD
        id_nsig   = _OD(
            (str(k), v)
            for k, v in sorted(road_cfg["id_nsig"].items(), key=lambda x: str(x[0]))
        )
        id_mps    = {str(k): v for k, v in road_cfg["id_mps"].items()}
        fixed_ids = sorted(id_nsig.keys())
        n_sigs    = sum(id_nsig.values())

        
        _FOCUSED = ["208", "1255", "1760"]
        focused_in_model = [cid for cid in _FOCUSED if cid in fixed_ids]

        print(f"  [CANet] ROAD: {len(fixed_ids)} IDs, {n_sigs} total signals")

        
        from tensorflow import keras
        road_n_sigs = n_sigs
        inputs_road = {
            cid: keras.Input(shape=(id_mps[cid], id_nsig[cid]), name=cid)
            for cid in fixed_ids
        }
        lstms_road = [
            keras.layers.LSTM(5 * id_nsig[cid], name=f"lstm_{cid}")
            for cid in fixed_ids
        ]
        x_road = [lstms_road[i](inputs_road[cid]) for i, cid in enumerate(fixed_ids)]
        x_road = keras.layers.Concatenate()(x_road)
        x_road = keras.layers.Dense((5 * road_n_sigs) // 2, activation="elu")(x_road)
        x_road = keras.layers.Dense(road_n_sigs - 1,         activation="elu")(x_road)
        out_road = keras.layers.Dense(road_n_sigs,            activation="elu")(x_road)
        self.model = keras.Model(inputs_road, out_road)

        
        models_subdir = os.path.join(project_root, "models", "ROAD")
        index_files   = sorted(glob.glob(os.path.join(models_subdir, "*.index")))
        if index_files:
            
            weights_ckpt = index_files[-1].replace(".index", "")
        elif self._weights_path:
            weights_ckpt = (
                self._weights_path[:-3]
                if self._weights_path.endswith(".h5")
                else self._weights_path
            )
        else:
            raise FileNotFoundError(
                "CANet ROAD: no weights found in models/ROAD/.\n"
                "Copy canet_road_epoch06.{index,data-*} there first."
            )
        self.model.load_weights(weights_ckpt)
        print(f"  [CANet] Loaded ROAD weights: {os.path.basename(weights_ckpt)}")

        
        
        
        _WINDOW_SIZE   = 1          
        _TIME_CUTOFF   = _WINDOW_SIZE + 1   
        _BATCH_SIZE    = 250

        def _choose_n_slices(file_path: Path) -> int:
            """
            Pick number of slices so each slice spans ≥ 3× the time cutoff.
            Prevents every slice from being shorter than the warm-up window
            (which would leave n_rows=0 for every slice and skip the file).
            Min slice duration target = 3 * _TIME_CUTOFF = 6 s.
            """
            try:
                df_head = pd.read_parquet(file_path, columns=["Time"])
                duration = float(df_head["Time"].max() - df_head["Time"].min())
            except Exception:
                return 1
            if duration <= 0:
                return 1
            
            n = max(1, int(duration / (3 * _TIME_CUTOFF)))
            return n

        
        def _slice_parquet(file_path: Path, n_slices: int) -> list[Path]:
            """
            Split a large parquet into n_slices roughly-equal pieces using
            pyarrow streaming.  This avoids loading the full file into RAM.
            Mirrors slice_parquet() from run_canet_road.py exactly.
            """
            cache_dir.mkdir(parents=True, exist_ok=True)
            pf = pq.ParquetFile(str(file_path))
            total = pf.metadata.num_rows
            slice_size = max(1, total // n_slices)
            boundaries = list(range(0, total, slice_size)) + [total]
            
            if len(boundaries) > 2 and (boundaries[-1] - boundaries[-2]) < slice_size // 2:
                boundaries.pop(-2)
            n_actual = len(boundaries) - 1
            paths: list[Path] = []
            slice_idx = 0
            buf: list[pa.RecordBatch] = []
            global_row = 0
            for batch in pf.iter_batches(batch_size=200_000):
                offset = 0
                while offset < batch.num_rows and slice_idx < n_actual:
                    next_boundary = boundaries[slice_idx + 1]
                    can_take = next_boundary - global_row
                    take = min(batch.num_rows - offset, can_take)
                    buf.append(batch.slice(offset, take))
                    offset += take
                    global_row += take
                    if global_row >= next_boundary:
                        p_out = cache_dir / f"{file_path.stem}_slice{slice_idx + 1}.parquet"
                        pq.write_table(pa.Table.from_batches(buf), str(p_out))
                        paths.append(p_out)
                        slice_idx += 1
                        buf = []
            if buf:
                p_out = cache_dir / f"{file_path.stem}_slice{slice_idx + 1}.parquet"
                pq.write_table(pa.Table.from_batches(buf), str(p_out))
                paths.append(p_out)
            return paths

        
        def _prepare_slice(
            file_path: Path,
        ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray] | None:
            """
            Load one parquet slice, build per-ID sliding windows, extract y.

            Returns None if there are no valid rows (e.g. too short for warm-up).

            The ROAD parquets have both 'Session' (time-window label, 0/1) and
            'Label' (exact injected-frame flag, 0/1).  We preserve both.
            time_and_labels columns: [Time, Session, Label]
            """
            df = pd.read_parquet(file_path)
            df.reset_index(drop=True, inplace=True)
            df["ID"] = df["ID"].astype(str)
            df = df.reset_index(names="Idx")

            time_start = df["Time"].iloc[0]
            mask       = df["Time"] > time_start + _TIME_CUTOFF
            n_rows     = int(mask.sum())
            if n_rows == 0:
                return None

            label_col = "Label" if "Label" in df.columns else "Session"
            tl = df.loc[mask, ["Time", "Session", label_col]].to_numpy()

            x_dict: dict[str, np.ndarray] = {}
            for cid in fixed_ids:
                n_sig  = id_nsig[cid]
                n_step = id_mps[cid]
                sig_cols = [f"Signal{i}" for i in range(1, n_sig + 1)]
                df_id = df.loc[df["ID"] == cid, ["Idx", "Session"] + sig_cols]
                if len(df_id) < n_step:
                    
                    x_dict[cid] = np.zeros((n_rows, n_step, n_sig), dtype=np.float32)
                    continue
                np_sig = df_id[["Session"] + sig_cols].to_numpy(dtype=np.float32)
                np_seq = np.lib.stride_tricks.sliding_window_view(
                    np_sig, window_shape=n_step, axis=0
                )
                np_seq = np_seq.swapaxes(1, 2)
                np_seq = np_seq[:, :, 1:]           
                n_seq  = np_seq.shape[0]
                end_idx   = df["Idx"].iloc[-1]
                n_repeats = np.diff(df_id["Idx"].to_list() + [end_idx])[-n_seq:]
                repeated  = np.repeat(np_seq, n_repeats, axis=0)
                x_dict[cid] = repeated[-n_rows:].copy()

            
            common = min([n_rows] + [v.shape[0] for v in x_dict.values()])
            if common <= 0:
                return None
            for cid in list(x_dict.keys()):
                x_dict[cid] = x_dict[cid][-common:].copy()
            tl = tl[-common:]

            y = np.concatenate([x_dict[cid][:, -1, :] for cid in fixed_ids], axis=1)
            return x_dict, y, tl

        
        def _run_file(file_path: Path) -> pd.DataFrame | None:
            """
            Slice file → inference on each slice → concat results.

            Returns a DataFrame with columns:
              Time, MSE, Session, Label, MSE_208, MSE_1255, MSE_1760
            Returns None if no valid slices exist.
            """
            slices = _slice_parquet(file_path, _choose_n_slices(file_path))
            parts: list[pd.DataFrame] = []

            for sl_path in slices:
                result = _prepare_slice(sl_path)
                if result is None:
                    continue
                x_dict, y_true, tl = result

                
                with tf.device("CPU"):
                    ds     = tf.data.Dataset.from_tensor_slices(x_dict)
                    ds     = ds.batch(_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
                    y_pred = self.model.predict(ds, verbose=0)

                
                err      = (y_true.astype(np.float32) - y_pred.astype(np.float32)) ** 2
                mse_vals = err.mean(axis=1)

                row_dict: dict[str, np.ndarray] = {
                    "Time":    tl[:, 0],
                    "MSE":     mse_vals,
                    "Session": tl[:, 1],
                    "Label":   tl[:, 2].astype(int),
                }

                
                
                
                col_start = 0
                for cid in fixed_ids:
                    n = id_nsig[cid]
                    col_end = col_start + n
                    if cid in focused_in_model:
                        row_dict[f"MSE_{cid}"] = err[:, col_start:col_end].mean(axis=1)
                    col_start = col_end

                part = pd.DataFrame(row_dict, dtype=float)
                part["Time"]    = part["Time"].round(6)
                part["Session"] = part["Session"].astype(int)
                part["Label"]   = part["Label"].astype(int)
                parts.append(part)
                del ds, y_pred, x_dict, y_true
                gc.collect()

            if not parts:
                return None
            return pd.concat(parts, axis=0, ignore_index=True)

        
        def _result_name(stem: str) -> str:
            if stem.startswith("valid_"):
                return f"ROAD_CANet_{stem}.parquet"
            if stem.startswith("test_"):
                attack_stem = stem.removeprefix("test_")
                return f"ROAD_CANet_{attack_stem}.parquet"
            return f"ROAD_CANet_{stem}.parquet"

        
        valid_files = sorted(canet_dir.glob("valid_*.parquet"))
        if not valid_files:
            raise FileNotFoundError(
                f"No valid_*.parquet files found in {canet_dir}.\n"
                "Run CanivalExtractor (Stage 1) first."
            )

        valid_mse_normal: list[np.ndarray] = []
        for vf in valid_files:
            print(f"  [CANet] ROAD valid: {vf.name}")
            vdf = _run_file(vf)
            if vdf is None:
                print(f"  [CANet] [WARN] {vf.name} produced no rows — skipping")
                continue
            
            out_name = _result_name(vf.stem)
            vdf.to_parquet(results_dir / out_name, index=False)
            
            normal_mse = vdf.loc[vdf["Session"] == 0, "MSE"].to_numpy()
            if len(normal_mse) > 0:
                valid_mse_normal.append(normal_mse)

        if not valid_mse_normal:
            raise RuntimeError(
                "CANet ROAD: no normal-traffic (Session==0) rows found in "
                "validation files. Cannot set threshold."
            )
        all_normal_mse = np.concatenate(valid_mse_normal)
        
        
        self.threshold = float(np.quantile(all_normal_mse, _CANET_VALID_QUANTILE))
        print(
            f"  [CANet] ROAD threshold (normal-row p99.9) = {self.threshold:.6e} "
            f"({len(all_normal_mse):,} normal rows)"
        )

        
        
        _ATTACK_ID_MAP = {
            "correlated_signal":  "MSE_1760",
            "max_speedometer":    "MSE_208",
            "reverse_light_off":  "MSE_208",
            "reverse_light_on":   "MSE_208",
            "max_engine_coolant": "MSE_1255",
        }
        _CANET_QUANTILE = 0.999    
        _CANET_LOW_Q    = 0.0001   

        def _adaptive_thr(df: pd.DataFrame, col: str, q: float) -> float:
            """p{q} of Session==0 rows; falls back to all rows if <100 normal rows."""
            normal = df[df["Session"] == 0] if int((df["Session"] == 0).sum()) >= 100 else df
            vals = normal[col].dropna()
            return float(np.quantile(vals, q)) if len(vals) else 0.0

        def _attack_family(stem: str) -> str:
            for kw in _ATTACK_ID_MAP:
                if kw in stem:
                    return kw
            return ""

        test_files = sorted(canet_dir.glob("test_*.parquet"))
        records: list[dict] = []

        for tf_file in test_files:
            stem     = tf_file.stem
            out_name = _result_name(stem)
            print(f"  [CANet] ROAD test:  {tf_file.name}")

            result_df = _run_file(tf_file)
            if result_df is None:
                print(f"  [CANet] [WARN] {tf_file.name} produced no rows — skipping")
                records.append({"file": tf_file.name, "result_file": out_name,
                                 "rows": 0, "status": "empty"})
                continue

            result_df.to_parquet(results_dir / out_name, index=False)

            
            
            
            family      = _attack_family(stem)
            focused_col = _ATTACK_ID_MAP.get(family, "")
            use_focused = (focused_col and focused_col in result_df.columns
                           and result_df[focused_col].sum() != 0)
            det_col  = focused_col if use_focused else "MSE"
            col_note = focused_col if use_focused else "GlobalMSE"

            thr_high = _adaptive_thr(result_df, det_col, _CANET_QUANTILE)
            thr_low  = _adaptive_thr(result_df, det_col, _CANET_LOW_Q)

            s_arr  = result_df["Session"].to_numpy().astype(int)
            mse_a  = result_df[det_col].fillna(0.0).to_numpy()
            binary = ((mse_a >= thr_high) | (mse_a <= thr_low)).astype(int)

            tn, fp, fn, tp, tpr, tnr, f1 = _evaluation(s_arr, binary)

            
            _W, _K = 200, 1
            n      = len(s_arr)
            n_win  = n // _W
            w_tpr = w_tnr = w_f1 = None
            if n_win > 0:
                lbl_mat = s_arr[:n_win * _W].reshape(n_win, _W)
                prd_mat = binary[:n_win * _W].reshape(n_win, _W)
                win_lbl = lbl_mat.max(axis=1)
                win_prd = (prd_mat.sum(axis=1) >= _K).astype(int)
                _, _, _, _, w_tpr, w_tnr, w_f1 = _evaluation(win_lbl, win_prd)

            rec = {
                "file":        tf_file.name,
                "result_file": out_name,
                "rows":        int(len(result_df)),
                "status":      "processed",
                "det_col":     col_note,
                "thr_high":    round(thr_high, 8),
                "thr_low":     round(thr_low, 8),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "TPR":   None if tpr   is None else float(tpr),
                "TNR":   float(tnr),
                "F1":    None if f1    is None else float(f1),
                "W_TPR": None if w_tpr is None else float(w_tpr),
                "W_TNR": None if w_tnr is None else float(w_tnr),
                "W_F1":  None if w_f1  is None else float(w_f1),
            }
            records.append(rec)
            w_str = (f"W-TPR={w_tpr or 0:.3f}  W-TNR={w_tnr or 0:.3f}  W-F1={w_f1 or 0:.3f}"
                     if w_tpr is not None else "")
            print(
                f"  [CANet] {stem:50s}  "
                f"TPR={tpr or 0:.3f}  TNR={tnr:.3f}  F1={f1 or 0:.3f}"
                + (f" │ {w_str}" if w_str else "")
                + f"  [{col_note}]"
            )

        
        output = {
            "dataset":   "ROAD",
            "model":     "CANet",
            "threshold": self.threshold,
            "n_ids":     len(fixed_ids),
            "n_sigs":    n_sigs,
            "by_file":   {r["file"]: r for r in records},
        }
        summary_path = results_dir / "ROAD_CANet_run_summary.json"
        with open(summary_path, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"  [CANet] ROAD summary → {summary_path}")
        return output

    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs) -> dict:
        """
        Run CANet inference on all eval files and compute per-attack metrics.
        Saves MSE result parquets to results/{dataset_name}/ for FusionIDS.
        """
        cfg = cfg or {}
        dataset_name = cfg.get("dataset_name", "SynCAN")

        if dataset_name == "SynCAN":
            return self._test_syncan(cfg)
        elif dataset_name == "X-CANIDS":
            return self._test_xcanids(cfg)
        elif dataset_name == "ROAD":
            return self._test_road(cfg)
        else:
            raise NotImplementedError(
                f"CANet.test() not yet implemented for dataset: {dataset_name}"
            )








_FUSION_SYNCAN_SPLITS = ["normal", "flooding", "suppress", "plateau", "continuous", "playback"]









_XCANIDS_CANET_QUANTILE = 0.9999  
_XCANIDS_TIL_QUANTILE   = 0.001   



_XCANIDS_VARIANT_RE = re.compile(r"_((?:fabr|fuzz|masq|repl|susp)-.+)$")









class FusionIDS(IDS):
    """
    OR-fusion of TIL and CANet detectors.
    Reads per-file result parquets saved by TIL.test() and CANet.test(),
    applies thresholds derived from the shared validation files, and
    computes per-attack and macro-averaged metrics.

    No trainable state — all logic is in test().
    Supports SynCAN, X-CANIDS, and ROAD via _test_syncan() / _test_xcanids() / _test_road().
    Dispatched by test(cfg={"dataset_name": ...}).
    """

    def __init__(self):
        pass

    

    def load(self, path: str):
        """FusionIDS has no trainable state — this is a no-op."""
        print("  [FusionIDS] No weights to load. Ready.")

    def save(self, path: str):
        """FusionIDS has no trainable state — nothing to save."""
        pass

    def train(self, train_dataset_dir=None, X_train=None, Y_train=None, cfg=None, **kwargs):
        """
        FusionIDS has no parameters of its own — thresholds are set
        individually from validation data inside test(). This is a no-op.
        """
        pass

    def predict(self, X_test=None, **kwargs) -> np.ndarray:
        """
        Given a DataFrame with 'z_itv' (TIL) and 'MSE' (CANet) columns
        plus 'til_threshold' and 'canet_threshold' kwargs, return OR-fused
        binary predictions.
        """
        til_threshold   = kwargs.get("til_threshold")
        canet_threshold = kwargs.get("canet_threshold")
        if til_threshold is None or canet_threshold is None:
            raise ValueError("FusionIDS.predict(): provide til_threshold and canet_threshold as kwargs.")
        if X_test is None:
            raise ValueError("FusionIDS.predict(): X_test must be a DataFrame with z_itv and MSE columns.")
        til_binary   = (X_test["z_itv"] < til_threshold).astype(int)
        canet_binary = (X_test["MSE"]   >= canet_threshold).astype(int)
        return np.maximum(til_binary, canet_binary)

    

    def _test_syncan(self, cfg: dict) -> dict:
        """
        SynCAN-specific fusion evaluation.
        Ported 1-to-1 from Deliverables/1_SynCAN/scripts/eval_syncan_fusion.py.
        """
        project_root = os.path.normpath(os.path.join(cfg["dir_path"], ".."))
        results_dir  = os.path.join(project_root, "results", "SynCAN")

        
        til_valid_path   = os.path.join(results_dir, "Syncan_TIL_valid.parquet")
        canet_valid_path = os.path.join(results_dir, "Syncan_CANet_valid.parquet")

        for p in (til_valid_path, canet_valid_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"FusionIDS: {p} not found. "
                    "Run TIL.test() and CANet.test() before FusionIDS.test()."
                )

        til_valid   = pd.read_parquet(til_valid_path)
        canet_valid = pd.read_parquet(canet_valid_path)

        thresholds = {
            "TIL":   float(np.quantile(til_valid["z_itv"], _TIL_VALID_QUANTILE)),
            "CANet": float(np.quantile(canet_valid["MSE"], _CANET_VALID_QUANTILE)),
        }
        print(f"  [Fusion] Thresholds — TIL: {thresholds['TIL']:.6f}  "
              f"CANet: {thresholds['CANet']:.6f}")

        
        eval_results: dict[str, dict[str, dict]] = {
            "TIL": {}, "CANet": {}, "Multimodal": {},
        }

        for attack in _FUSION_SYNCAN_SPLITS:
            til_path   = os.path.join(results_dir, f"Syncan_TIL_{attack}.parquet")
            canet_path = os.path.join(results_dir, f"Syncan_CANet_{attack}.parquet")

            if not os.path.exists(til_path) or not os.path.exists(canet_path):
                print(f"  [Fusion] [WARN] Missing result files for '{attack}' — skipping")
                continue

            til_df   = pd.read_parquet(til_path).sort_values("Time").reset_index(drop=True)
            canet_df = pd.read_parquet(canet_path).sort_values("Time").reset_index(drop=True)

            
            til_df["TIL"]     = (til_df["z_itv"]   <  thresholds["TIL"]).astype(int)
            canet_df["CANet"] = (canet_df["MSE"]    >= thresholds["CANet"]).astype(int)

            
            
            
            
            t_start = max(float(til_df["Time"].iloc[0]),   float(canet_df["Time"].iloc[0]))
            t_end   = min(float(til_df["Time"].iloc[-1]),  float(canet_df["Time"].iloc[-1]))
            til_df   = til_df.query("@t_start <= Time <= @t_end").reset_index(drop=True)
            canet_df = canet_df.query("@t_start <= Time <= @t_end").reset_index(drop=True)

            
            
            
            
            merged = pd.merge(
                til_df[["Time", "Session", "TIL"]],
                canet_df[["Time", "Session", "CANet"]],
                left_index=True, right_index=True,
            )
            
            merged["Multimodal"] = merged[["TIL", "CANet"]].max(axis=1)

            
            
            
            labels = merged["Session_y"].to_numpy()

            for model_name, col in [
                ("TIL",        "TIL"),
                ("CANet",      "CANet"),
                ("Multimodal", "Multimodal"),
            ]:
                tn, fp, fn, tp, tpr, tnr, f1 = _evaluation(labels, merged[col].to_numpy())
                eval_results[model_name][attack] = {
                    "rows": int(len(merged)),
                    "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                    "TPR": None if tpr is None else float(tpr),
                    "TNR": float(tnr),
                    "F1":  None if f1  is None else float(f1),
                }

            mm = eval_results["Multimodal"][attack]
            print(f"  [Fusion] {attack:12s}  "
                  f"TPR={mm['TPR'] or 0:.3f}  TNR={mm['TNR']:.3f}  F1={mm['F1'] or 0:.3f}")

        
        macro: dict[str, dict] = {}
        for model_name, by_attack in eval_results.items():
            tprs = [v["TPR"] for v in by_attack.values() if v["TPR"] is not None]
            tnrs = [v["TNR"] for v in by_attack.values() if v["TNR"] is not None]
            f1s  = [v["F1"]  for v in by_attack.values() if v["F1"]  is not None]
            macro[model_name] = {
                "TPR": float(np.mean(tprs)) if tprs else None,
                "TNR": float(np.mean(tnrs)) if tnrs else None,
                "F1":  float(np.mean(f1s))  if f1s  else None,
            }

        output = {
            "dataset":    "SynCAN",
            "scope":      "attack-aligned SynCAN evaluation",
            "thresholds": thresholds,
            "by_attack":  eval_results,
            "macro":      macro,
        }
        out_path = os.path.join(results_dir, "Syncan_fusion_metrics.json")
        with open(out_path, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"  [Fusion] Metrics → {out_path}")
        return output

    

    def _test_xcanids(self, cfg: dict) -> dict:
        """
        X-CANIDS-specific fusion evaluation.
        Ported 1-to-1 from Deliverables/2_X-CANIDS/scripts/eval_xcanids_fusion.py.

        Key differences from SynCAN:
          - Thresholds: CANet=0.9999th percentile, TIL=0.001th percentile
          - Alignment: inner join on Time column (not t_start/t_end window)
          - Labels: Session != 0 (not just == 1)
          - Aggregation: by variant AND by attack family
        """
        project_root = os.path.normpath(os.path.join(cfg["dir_path"], ".."))
        results_dir  = os.path.join(project_root, "results", "X-CANIDS")

        
        canet_valid_path = os.path.join(results_dir, "X-CANIDS_CANet_valid.parquet")
        til_valid_path   = os.path.join(results_dir, "X-CANIDS_TIL_valid.parquet")
        for p in (canet_valid_path, til_valid_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"FusionIDS X-CANIDS: {p} not found. "
                    "Run CANet.test() and TIL.test() before FusionIDS.test()."
                )

        canet_valid = pd.read_parquet(canet_valid_path)
        til_valid   = pd.read_parquet(til_valid_path)
        thr_canet   = float(np.quantile(canet_valid["MSE"],   _XCANIDS_CANET_QUANTILE))
        thr_til     = float(np.quantile(til_valid["z_itv"],   _XCANIDS_TIL_QUANTILE))
        print(f"  [Fusion] X-CANIDS thresholds — CANet: {thr_canet:.6e}  TIL: {thr_til:.8f}")

        
        def _variant_key(path: str) -> str | None:
            """
            Extract the variant tag from a result filename.
            e.g. "X-CANIDS_CANet_fabr-001.parquet" → "fabr-001"
            Returns None if the filename doesn't match the expected pattern.
            """
            m = _XCANIDS_VARIANT_RE.search(os.path.splitext(os.path.basename(path))[0])
            return m.group(1) if m else None

        canet_map = {
            _variant_key(p): p
            for p in sorted(
                f for f in os.listdir(results_dir)
                if f.startswith("X-CANIDS_CANet_") and f.endswith(".parquet")
                and not f.endswith("valid.parquet")
            )
            if _variant_key(p)
        }
        til_map = {
            _variant_key(p): p
            for p in sorted(
                f for f in os.listdir(results_dir)
                if f.startswith("X-CANIDS_TIL_") and f.endswith(".parquet")
                and not f.endswith("valid.parquet")
            )
            if _variant_key(p)
        }

        common_keys = sorted(set(canet_map) & set(til_map))
        print(f"  [Fusion] X-CANIDS matched variant pairs: {len(common_keys)}")

        
        by_variant: dict[str, dict] = {}
        by_family:  dict[str, dict] = {}

        for key in common_keys:
            df_c = pd.read_parquet(
                os.path.join(results_dir, canet_map[key]),
                columns=["Time", "MSE", "Session"],
            )
            df_t = pd.read_parquet(
                os.path.join(results_dir, til_map[key]),
                columns=["Time", "z_itv", "Session"],
            )

            
            merged = df_c.merge(df_t[["Time", "z_itv"]], on="Time", how="inner")
            if len(merged) == 0:
                print(f"  [Fusion] [WARN] No matching timestamps for {key} — skipping")
                continue

            
            labels      = (merged["Session"] != 0).astype(int).values
            canet_pred  = (merged["MSE"]   >= thr_canet).astype(int).values
            til_pred    = (merged["z_itv"] <  thr_til).astype(int).values
            fusion_pred = np.maximum(canet_pred, til_pred)

            def _bm(lbl, prd):
                """
                Compute TP/TN/FP/FN + TPR/TNR/F1 for a single variant.
                Uses precision-based F1 (2*P*R/(P+R)) to match
                eval_xcanids_fusion.py — differs from the SynCAN formula
                (2*TP / (2*TP+FP+FN)) but produces the same value.
                Returns 0.0 for undefined metrics (no positives/negatives).
                """
                tp = int(((prd == 1) & (lbl == 1)).sum())
                tn = int(((prd == 0) & (lbl == 0)).sum())
                fp = int(((prd == 1) & (lbl == 0)).sum())
                fn = int(((prd == 0) & (lbl == 1)).sum())
                tpr  = tp / (tp + fn) if (tp + fn) else 0.0
                tnr  = tn / (tn + fp) if (tn + fp) else 0.0
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                f1   = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
                return dict(tp=tp, tn=tn, fp=fp, fn=fn, TPR=tpr, TNR=tnr, F1=f1)

            by_variant[key] = {
                "n_rows": int(len(merged)),
                "n_pos":  int(labels.sum()),
                "n_neg":  int((labels == 0).sum()),
                "TIL":    _bm(labels, til_pred),
                "CANet":  _bm(labels, canet_pred),
                "Fusion": _bm(labels, fusion_pred),
            }

            fam = key.split("-")[0]
            if fam not in by_family:
                by_family[fam] = {"TIL": [], "CANet": [], "Fusion": []}
            for m in ("TIL", "CANet", "Fusion"):
                by_family[fam][m].append(by_variant[key][m])

            fus = by_variant[key]["Fusion"]
            print(f"  [Fusion] {key:30s}  F1={fus['F1']:.3f}  "
                  f"TPR={fus['TPR']:.3f}  TNR={fus['TNR']:.3f}")

        
        def _macro(metric_list: list[dict]) -> dict:
            """
            Macro-average TPR, TNR, F1 across a list of per-variant metric dicts.
            Each variant gets equal weight regardless of its size (number of rows).
            Mirrors the aggregation in eval_xcanids_fusion.py.
            """
            return {
                k: float(np.mean([m[k] for m in metric_list]))
                for k in ("TPR", "TNR", "F1")
            }

        family_summary: dict[str, dict] = {
            fam: {m: _macro(by_family[fam][m]) for m in ("TIL", "CANet", "Fusion")}
            for fam in sorted(by_family)
        }

        all_til    = [by_variant[k]["TIL"]    for k in by_variant]
        all_canet  = [by_variant[k]["CANet"]  for k in by_variant]
        all_fusion = [by_variant[k]["Fusion"] for k in by_variant]
        overall = {
            "TIL":    _macro(all_til)    if all_til    else {},
            "CANet":  _macro(all_canet)  if all_canet  else {},
            "Fusion": _macro(all_fusion) if all_fusion else {},
        }

        print("\n  [Fusion] X-CANIDS OVERALL Fusion →  "
              f"TPR={overall['Fusion'].get('TPR', 0):.3f}  "
              f"TNR={overall['Fusion'].get('TNR', 0):.3f}  "
              f"F1={overall['Fusion'].get('F1', 0):.3f}")

        output = {
            "dataset":    "X-CANIDS",
            "thresholds": {"CANet": thr_canet, "TIL": thr_til},
            "by_variant": by_variant,
            "by_family":  family_summary,
            "overall":    overall,
        }
        out_path = os.path.join(results_dir, "X-CANIDS_fusion_metrics.json")
        with open(out_path, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"  [Fusion] X-CANIDS metrics → {out_path}")
        return output

    

    def _test_road(self, cfg: dict) -> dict:
        """
        ROAD-specific fusion evaluation.

        Ported 1-to-1 from Deliverables/3_ROAD/scripts/eval_road_fusion.py.

        Key ROAD-specific design choices vs SynCAN/X-CANIDS:
          - Adaptive thresholds: computed per-file from normal rows (Session==0)
            of each test file, NOT a single global validation threshold.
            Falls back to validation parquets when < MIN_NORMAL_TIL_ROWS rows.
          - Focused detectors: uses per-ID MSE (MSE_208 / MSE_1255 / MSE_1760)
            and per-ID z_itv (z_itv_208 / z_itv_1255 / z_itv_1760) when the
            attacked CAN ID is known for that family.  Falls back to global
            MSE / z_itv otherwise.
          - Bidirectional CANet: flags a row if MSE is too HIGH (anomaly) OR too
            LOW (reconstruction collapse), mirroring the masquerade-attack pattern.
          - Time alignment: merge_asof (tolerance 1e-3 s) rather than exact
            join or index-merge, because TIL and CANet operate on different
            message subsets so timestamps may not match exactly.
          - Ground truth: Label (exact injected frames) when available;
            falls back to Session (time-window label).
          - Window-level evaluation: W=200 rows per window, K=1 min flagged
            rows inside a window for the window to count as a detection.
          - Grouping: files are grouped by (attack_family, variant_type) where
            variant_type is "injection" or "masquerade".
        """
        import re as _re
        from pathlib import Path as _P

        project_root = os.path.normpath(os.path.join(cfg["dir_path"], ".."))
        results_dir  = _P(project_root) / "results" / "ROAD"

        if not results_dir.exists():
            raise FileNotFoundError(
                f"FusionIDS ROAD: results directory {results_dir} not found.\n"
                "Run CANet.test() and TIL.test() before FusionIDS.test()."
            )

        
        _CANET_QUANTILE   = 0.999    
        _CANET_LOW_Q      = 0.0001   # p0.01 of normal MSE → lower threshold (bidirectional)
        _TIL_QUANTILE     = 0.001    # p0.1  of normal z_itv → lower threshold
        _WINDOW_W         = 200      # window size (rows) for window-level eval
        _WINDOW_K         = 1        # min flagged rows per window to call it detected (K=1 = "any row flagged")
        _MIN_NORMAL_TIL   = 200      # min normal TIL rows before falling back to valid

        
        _ATTACK_ID_MAP: dict[str, str] = {
            "correlated_signal":  "1760",
            "max_speedometer":    "208",
            "reverse_light_off":  "208",
            "reverse_light_on":   "208",
            "max_engine_coolant": "1255",
        }
        
        _FOCUSED_TIL_COL: dict[str, str] = {
            "1760": "z_itv_1760",
            "208":  "z_itv_208",
            "1255": "z_itv_1255",
        }
        _NO_LABEL_KEYS = {"accelerator", "fuzzing"}

        

        def _binary_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
            """Row-level TP/TN/FP/FN + TPR/TNR/F1 with epsilon denominator."""
            tp = int(((preds == 1) & (labels == 1)).sum())
            tn = int(((preds == 0) & (labels == 0)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            fn = int(((preds == 0) & (labels == 1)).sum())
            tpr = tp / (tp + fn + 1e-9)
            tnr = tn / (tn + fp + 1e-9)
            f1  = (2 * tp) / (2 * tp + fp + fn + 1e-9)
            return {"TPR": round(tpr, 4), "TNR": round(tnr, 4), "F1": round(f1, 4),
                    "TP": tp, "TN": tn, "FP": fp, "FN": fn}

        def _window_metrics(
            labels: np.ndarray, preds: np.ndarray, W: int, K: int
        ) -> dict:
            """
            Window-level evaluation: slice into W-row windows, call a window
            as positive if ≥ K rows inside are flagged.
            Mirrors window_metrics() in eval_road_fusion.py.
            """
            n = len(labels)
            n_win = n // W
            if n_win == 0:
                return {"TPR": 0.0, "TNR": 0.0, "F1": 0.0,
                        "n_windows": 0, "n_pos": 0, "n_neg": 0}
            lbl_mat = labels[: n_win * W].reshape(n_win, W)
            prd_mat = preds [: n_win * W].reshape(n_win, W)
            win_lbl = lbl_mat.max(axis=1)
            win_prd = (prd_mat.sum(axis=1) >= K).astype(int)
            m = _binary_metrics(win_lbl, win_prd)
            m["n_windows"] = n_win
            m["n_pos"] = int(win_lbl.sum())
            m["n_neg"] = int((win_lbl == 0).sum())
            return m

        def _attack_family_and_variant(stem: str) -> tuple[str, str]:
            """
            Parse attack family and variant type from a result parquet stem.
            e.g. "ROAD_CANet_correlated_signal_1_masquerade" →
                     family="correlated_signal", variant="masquerade"
            Mirrors attack_family_and_variant() in eval_road_fusion.py.
            """
            
            clean = _re.sub(r"^ROAD_(CANet|TIL_focused|TIL)_(valid_)?", "", stem)
            clean = clean.removesuffix(".parquet")
            if "fuzzing" in clean:
                return "fuzzing", "fuzzing"
            if "accelerator" in clean:
                return "accelerator", "accelerator"
            is_masq = "_masquerade" in clean
            vtype   = "masquerade" if is_masq else "injection"
            for kw in ("correlated_signal", "max_speedometer", "reverse_light_off",
                       "reverse_light_on", "max_engine_coolant"):
                if kw in clean:
                    return kw, vtype
            return clean.replace("_masquerade", ""), vtype

        def _grouped_key(family: str, vtype: str) -> str:
            if vtype in ("fuzzing", "accelerator"):
                return vtype
            return f"{family}_{vtype}"

        def _adaptive_threshold(df: pd.DataFrame, col: str, q: float,
                                 fallback_df: pd.DataFrame | None) -> float:
            """
            Compute threshold from normal rows (Session==0) of this file.

            KEY: use only PRE-ATTACK Session==0 rows (rows before the first
            Session==1 row), not all Session==0 rows.  Post-attack Session==0
            rows in the recovery window (the n_step/rate seconds after the
            attack ends) still have elevated MSE because the sliding window
            still contains attack-period signal values.  Including them inflates
            the threshold to the attack MSE level, causing TPR→0.

            The original eval_road_fusion.py avoids this by accident: its fixed
            10-slice approach places the last slice boundary just past the
            recovery window, so post-attack rows only appear from pure
            post-recovery time onwards.  We replicate the intent explicitly here.

            If fallback_df is provided, use the validation pool instead.
            """
            if fallback_df is not None:
                fb_normal = fallback_df[fallback_df["Session"] == 0]
                fb_vals   = fb_normal[col].dropna()
                if len(fb_vals) > 0:
                    return float(np.quantile(fb_vals, q))

            
            attack_rows = df[df["Session"] == 1]
            if "Time" in df.columns and len(attack_rows) > 0:
                t_attack_start = float(attack_rows["Time"].min())
                pre_attack = df[(df["Session"] == 0) & (df["Time"] < t_attack_start)]
            else:
                pre_attack = df[df["Session"] == 0]

            if len(pre_attack) >= 100:
                vals = pre_attack[col].dropna()
            elif int((df["Session"] == 0).sum()) >= 100:
                
                vals = df[df["Session"] == 0][col].dropna()
            else:
                vals = df[col].dropna()

            return float(np.quantile(vals, q)) if len(vals) else 0.0

        
        canet_files = sorted(results_dir.glob("ROAD_CANet_*.parquet"))
        til_files   = sorted(results_dir.glob("ROAD_TIL_focused_*.parquet"))

        canet_valid_files = [f for f in canet_files if "_valid_" in f.stem]
        til_valid_files   = [f for f in til_files   if "_valid_" in f.stem]
        canet_test_files  = [f for f in canet_files if "_valid_" not in f.stem]
        til_test_files    = [f for f in til_files   if "_valid_" not in f.stem]

        if not canet_test_files:
            raise FileNotFoundError(
                f"FusionIDS ROAD: no ROAD_CANet_*.parquet attack files in {results_dir}.\n"
                "Run CANet.test() first."
            )
        if not til_test_files:
            raise FileNotFoundError(
                f"FusionIDS ROAD: no ROAD_TIL_focused_*.parquet attack files in {results_dir}.\n"
                "Run TIL.test() first."
            )

        print(f"  [Fusion] ROAD CANet test files : {len(canet_test_files)}")
        print(f"  [Fusion] ROAD TIL   test files : {len(til_test_files)}")
        print(f"  [Fusion] ROAD CANet valid files: {len(canet_valid_files)}")
        print(f"  [Fusion] ROAD TIL   valid files: {len(til_valid_files)}")

        
        til_valid_pool: pd.DataFrame | None = None
        if til_valid_files:
            til_valid_pool = pd.concat(
                [pd.read_parquet(f) for f in til_valid_files], ignore_index=True
            )

        
        canet_map: dict[str, list] = {}
        for f in canet_test_files:
            fam, vtype = _attack_family_and_variant(f.stem)
            canet_map.setdefault(_grouped_key(fam, vtype), []).append(f)

        til_map: dict[str, list] = {}
        for f in til_test_files:
            fam, vtype = _attack_family_and_variant(f.stem)
            til_map.setdefault(_grouped_key(fam, vtype), []).append(f)

        common_keys = sorted(set(canet_map) & set(til_map))
        print(f"  [Fusion] ROAD matched attack groups: {len(common_keys)}")
        if not common_keys:
            raise RuntimeError(
                "FusionIDS ROAD: CANet and TIL result files do not share any "
                "attack groups.  Check naming conventions."
            )

        
        all_results: dict[str, list[dict]] = {}

        for key in common_keys:
            is_masq    = key.endswith("_masquerade")
            is_no_lbl  = any(k in key for k in _NO_LABEL_KEYS)
            family     = key.rsplit("_injection", 1)[0].rsplit("_masquerade", 1)[0]
            attacked_id = _ATTACK_ID_MAP.get(family)

            
            canet_id_col   = f"MSE_{attacked_id}"   if attacked_id else None
            til_focused_col = _FOCUSED_TIL_COL.get(attacked_id) if attacked_id else None

            c_paths = sorted(canet_map[key])
            t_paths = sorted(til_map[key])
            n_variants = min(len(c_paths), len(t_paths))

            variant_results: list[dict] = []

            for i in range(n_variants):
                df_c = pd.read_parquet(c_paths[i]).sort_values("Time").reset_index(drop=True)
                df_t = pd.read_parquet(t_paths[i]).sort_values("Time").reset_index(drop=True)

                n_normal_canet = int((df_c["Session"] == 0).sum())
                n_normal_til   = int((df_t["Session"] == 0).sum())

                
                use_focused_canet = (
                    canet_id_col is not None
                    and canet_id_col in df_c.columns
                    and float(df_c[canet_id_col].sum()) != 0.0
                )
                canet_det_col  = canet_id_col if use_focused_canet else "MSE"
                canet_det_note = (f"MSE_ID{attacked_id}" if use_focused_canet
                                  else "GlobalMSE")

                use_focused_til = (
                    til_focused_col is not None
                    and til_focused_col in df_t.columns
                    and not df_t[til_focused_col].isna().all()
                )
                til_det_col  = til_focused_col if use_focused_til else "z_itv"
                til_det_note = (f"z_itv_ID{attacked_id}" if use_focused_til
                                else "z_itv_global")

                if not use_focused_til and not is_no_lbl:
                    print(f"  [Fusion] [WARN] {t_paths[i].name}: "
                          f"focused TIL col {til_focused_col} missing/NaN → global z_itv")

                
                
                thr_canet_high = _adaptive_threshold(df_c, canet_det_col,
                                                      _CANET_QUANTILE, None)
                thr_canet_low  = _adaptive_threshold(df_c, canet_det_col,
                                                      _CANET_LOW_Q, None)
                if n_normal_canet < 100:
                    print(f"  [Fusion] [WARN] {c_paths[i].name}: "
                          f"only {n_normal_canet} normal CANet rows.")

                
                til_fallback = False
                if n_normal_til < _MIN_NORMAL_TIL:
                    fb = til_valid_pool
                    til_fallback = True
                    print(f"  [Fusion] [V2-FALLBACK] {t_paths[i].name}: "
                          f"{n_normal_til} normal TIL rows → validation fallback")
                else:
                    fb = None
                thr_til = _adaptive_threshold(df_t, til_det_col, _TIL_QUANTILE, fb)

                
                
                high_flag = df_c[canet_det_col] >= thr_canet_high
                low_flag  = df_c[canet_det_col] <= thr_canet_low
                df_c["CANet_pred"] = (high_flag | low_flag).astype(int)

                
                df_t["TIL_pred"] = (df_t[til_det_col] < thr_til).astype(int)

                
                
                
                
                canet_cols = ["Time", "Session", "CANet_pred"]
                if "Label" in df_c.columns:
                    canet_cols.append("Label")

                merged = pd.merge_asof(
                    df_t[["Time", "Session", "TIL_pred"]].sort_values("Time"),
                    df_c[canet_cols].sort_values("Time"),
                    on="Time",
                    suffixes=("_til", "_canet"),
                    tolerance=1e-3,
                    direction="nearest",
                ).dropna()

                if len(merged) == 0:
                    print(f"  [Fusion] [WARN] No overlapping rows for {key} "
                          f"variant {i+1} — skipping.")
                    continue

                
                merged["Fusion_pred"] = merged[["TIL_pred", "CANet_pred"]].max(axis=1)

                
                
                if "Label" in merged.columns and not is_no_lbl:
                    labels     = merged["Label"].to_numpy().astype(int)
                    gt_source  = "Label"
                else:
                    labels     = merged["Session_canet"].to_numpy().astype(int)
                    gt_source  = "Session"

                
                msg_canet  = _binary_metrics(labels, merged["CANet_pred"].to_numpy())
                msg_til    = _binary_metrics(labels, merged["TIL_pred"].to_numpy())
                msg_fusion = _binary_metrics(labels, merged["Fusion_pred"].to_numpy())

                
                win_canet  = _window_metrics(labels, merged["CANet_pred"].to_numpy(),
                                              _WINDOW_W, _WINDOW_K)
                win_til    = _window_metrics(labels, merged["TIL_pred"].to_numpy(),
                                              _WINDOW_W, _WINDOW_K)
                win_fusion = _window_metrics(labels, merged["Fusion_pred"].to_numpy(),
                                              _WINDOW_W, _WINDOW_K)

                variant_rec = {
                    "variant":            c_paths[i].name,
                    "variant_type":       ("masquerade" if is_masq else
                                          ("fuzzing" if "fuzzing" in key else
                                           ("accelerator" if "accelerator" in key
                                            else "injection"))),
                    "ground_truth":       gt_source,
                    "canet_det_col":      canet_det_note,
                    "til_det_col":        til_det_note,
                    "til_fallback_used":  til_fallback,
                    "n_rows":             int(len(merged)),
                    "n_injected":         int(labels.sum()),
                    "n_normal_til_rows":  n_normal_til,
                    "thresholds": {
                        "canet_high": round(thr_canet_high, 8),
                        "canet_low":  round(thr_canet_low,  8),
                        "til":        round(thr_til,         8),
                    },
                    "msg":  {"CANet": msg_canet,  "TIL": msg_til,  "Fusion": msg_fusion},
                    "win":  {"CANet": win_canet,  "TIL": win_til,  "Fusion": win_fusion},
                }
                variant_results.append(variant_rec)

                fus_w = win_fusion
                print(
                    f"  [Fusion] {key:45s} v{i+1:02d}  "
                    f"W-F1={fus_w['F1']:.3f}  "
                    f"W-TPR={fus_w['TPR']:.3f}  "
                    f"W-TNR={fus_w['TNR']:.3f}"
                )

            if variant_results:
                all_results[key] = variant_results

    
        def _mean_win(keys: list[str]) -> dict:
            means: dict = {}
            for k in keys:
                variants = all_results.get(k, [])
                if not variants:
                    continue
                means[k] = {}
                for model in ("CANet", "TIL", "Fusion"):
                    wins = [v["win"].get(model, {}) for v in variants
                            if v["win"].get(model)]
                    if not wins:
                        continue
                    arr = np.array([[w["TPR"], w["TNR"], w["F1"]] for w in wins])
                    avg = arr.mean(axis=0)
                    means[k][model] = {
                        "TPR": round(float(avg[0]), 4),
                        "TNR": round(float(avg[1]), 4),
                        "F1":  round(float(avg[2]), 4),
                    }
            return means

        inj_keys  = sorted(k for k in all_results if k.endswith("_injection"))
        masq_keys = sorted(k for k in all_results if k.endswith("_masquerade"))
        other_keys = sorted(k for k in all_results
                             if k not in inj_keys + masq_keys)

        mean_win = {
            "injection":  _mean_win(inj_keys),
            "masquerade": _mean_win(masq_keys),
            "other":      _mean_win(other_keys),
        }

        
        all_fusion_wins = [
            v["win"]["Fusion"]
            for vlist in all_results.values()
            for v in vlist
            if v["win"].get("Fusion")
        ]
        if all_fusion_wins:
            overall_arr = np.array([[w["TPR"], w["TNR"], w["F1"]]
                                     for w in all_fusion_wins])
            overall_avg = overall_arr.mean(axis=0)
            overall = {"TPR": round(float(overall_avg[0]), 4),
                       "TNR": round(float(overall_avg[1]), 4),
                       "F1":  round(float(overall_avg[2]), 4)}
        else:
            overall = {}

        SEP = "=" * 130
        def _print_win_table(keys: list[str], title: str) -> None:
            if not keys:
                return
            print(f"\n{SEP}")
            print(f"  {title}  [W={_WINDOW_W}  K={_WINDOW_K}]")
            print(SEP)
            print(f"  {'Attack':<42} {'V':<3} {'Model':<8}  TPR    TNR    F1   "
                  f"#Win   #Pos  #Neg  TIL_col")
            print("-" * 130)
            for key in keys:
                for vi, vr in enumerate(all_results.get(key, [])):
                    for model in ("CANet", "TIL", "Fusion"):
                        w = vr["win"].get(model, {})
                        if not w:
                            continue
                        first = key if (vi == 0 and model == "CANet") else ""
                        var   = str(vi + 1) if model == "CANet" else ""
                        tcol  = vr["til_det_col"] if model == "TIL" else ""
                        fb    = "✓val" if (model == "TIL" and vr.get("til_fallback_used")) else ""
                        print(f"  {first:<42} {var:<3} {model:<8}"
                              f"  {w.get('TPR',0):.3f}  {w.get('TNR',0):.3f}  {w.get('F1',0):.3f}"
                              f"  {w.get('n_windows',0):>5}  {w.get('n_pos',0):>4}"
                              f"  {w.get('n_neg',0):>4}  {tcol:<22} {fb}")
                    print()

        inj_keys2  = sorted(k for k in all_results if k.endswith("_injection"))
        masq_keys2 = sorted(k for k in all_results if k.endswith("_masquerade"))
        other_keys2 = sorted(k for k in all_results
                              if k not in inj_keys2 + masq_keys2)

        _print_win_table(inj_keys2,  "WINDOW-LEVEL — INJECTION attacks")
        _print_win_table(masq_keys2, "WINDOW-LEVEL — MASQUERADE attacks")
        _print_win_table(other_keys2,"WINDOW-LEVEL — OTHER (fuzzing / accelerator)")

        
        print(f"\n{SEP}")
        print(f"  OVERALL MEAN — W={_WINDOW_W}, K={_WINDOW_K} — Injection vs Masquerade")
        print("-" * 90)
        for label2, mdict in [("INJECTION", _mean_win(inj_keys2)),
                               ("MASQUERADE", _mean_win(masq_keys2))]:
            for model in ("CANet", "TIL", "Fusion"):
                all_vals = [mdict[k][model] for k in mdict if model in mdict[k]]
                if not all_vals:
                    continue
                avg = np.array([[v["TPR"], v["TNR"], v["F1"]]
                                 for v in all_vals]).mean(axis=0)
                print(f"  {label2:<15} {model:<8}  "
                      f"TPR={avg[0]:.3f}  TNR={avg[1]:.3f}  F1={avg[2]:.3f}")
            print()

        print(f"\n  [Fusion] ROAD OVERALL Fusion (window W={_WINDOW_W}, K={_WINDOW_K}) → "
              f"TPR={overall.get('TPR', 0):.3f}  "
              f"TNR={overall.get('TNR', 0):.3f}  "
              f"F1={overall.get('F1', 0):.3f}")

        output = {
            "dataset": "ROAD",
            "settings": {
                "canet_quantile":    _CANET_QUANTILE,
                "canet_low_q":       _CANET_LOW_Q,
                "til_quantile":      _TIL_QUANTILE,
                "window_W":          _WINDOW_W,
                "window_K":          _WINDOW_K,
                "min_normal_til":    _MIN_NORMAL_TIL,
                "bidirectional_canet": True,
                "use_label":          True,
            },
            "road_attack_id_map": _ATTACK_ID_MAP,
            "per_group":          all_results,
            "mean_win":           mean_win,
            "overall_fusion_win": overall,
        }
        out_path = results_dir / "ROAD_fusion_metrics.json"
        with open(str(out_path), "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"  [Fusion] ROAD metrics → {out_path}")
        return output

    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs) -> dict:
        """
        Read TIL + CANet result parquets, derive thresholds from validation,
        OR-combine, compute per-attack and macro metrics, save fusion JSON.
        """
        cfg = cfg or {}
        dataset_name = cfg.get("dataset_name", "SynCAN")

        if dataset_name == "SynCAN":
            return self._test_syncan(cfg)
        elif dataset_name == "X-CANIDS":
            return self._test_xcanids(cfg)
        elif dataset_name == "ROAD":
            return self._test_road(cfg)
        else:
            raise NotImplementedError(
                f"FusionIDS.test() not yet implemented for dataset: {dataset_name}"
            )


class CANival(IDS):


    def __init__(self):
        self.til = TIL()
        self.canet = CANet()
        self.fusion = FusionIDS()
        self.results: dict = {}

    def load(self, path: str):
        
        
        self.canet.load(path)

    def save(self, path: str):
        self.canet.save(path)

    def train(self, train_dataset_dir=None, X_train=None, Y_train=None, cfg=None, **kwargs):
        self.canet.train(train_dataset_dir, X_train, Y_train, cfg=cfg, **kwargs)

    def predict(self, X_test=None, **kwargs):
        return self.fusion.predict(X_test, **kwargs)

    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs) -> dict:
        cfg = cfg or {}
        print("\n[CANival 1/3] TIL")
        self.results["TIL"] = self.til.test(cfg=cfg)
        print("\n[CANival 2/3] CANet")
        self.results["CANet"] = self.canet.test(cfg=cfg)
        print("\n[CANival 3/3] FusionIDS")
        self.results["Fusion"] = self.fusion.test(cfg=cfg)
        return self.results
