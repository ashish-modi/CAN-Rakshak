"""
CanditoExtractor — Feature extractor for the Candito IDS
=========================================================
Paper: S. Longari et al., "CANdito: Improving Payload-Based Detection of
       Attacks on Controller Area Networks", CSCML 2023.

Parses a raw `modified_dataset` CSV (timestamp, can_id, dlc,
byte0..byte7, R/T flag into the DataFrame shape CANdito's own algorithm expects: Time, Id,
Dlc, Payload (bitstring), IsTampered. This is only Step 0 of CANdito's
Module 1 (payload -> bits); it deliberately does NOT run READ signal
discovery or per-ID encoding here.

Why not: READ must be fit once on benign TRAINING data only, and every
other file (the test/attack file) must be encoded with THAT SAME fitted
signal set, not its own independently re-discovered one (paper Sec. 4.1).
Since this pipeline's dataset_processing stage runs once per configured
`file_name`, with train/test using different file_names across separate
runs, fitting READ here would silently re-discover different signal
boundaries for train vs. test data. So this extractor only caches the
parsed+bitified DataFrame; ids/candito.py fits READ during train() and
persists it via save()/load() for test() to reuse, mirroring how
CANShield's own SynCAN mode persists its train-only normalization stats
for reuse at test time.

Fully self-contained (no import from CANdito_IDS/): this pipeline has no
runtime dependency on that folder.
"""

import os

import pandas as pd

from features.feature_extractors.base import FeatureExtractor


def _normalize_payload_column(df):
    """Ensure the 'Payload' column is a binary string of length Dlc*8.

    Accepts hex or already-binary payload text; rows whose payload can't be
    parsed are dropped.
    """
    df = df.copy()

    def to_bits(row):
        p = str(row["Payload"]).strip()
        nbits = int(row["Dlc"]) * 8
        if set(p) <= {"0", "1"} and len(p) in (nbits, 0):
            return p.zfill(nbits)
        try:
            return bin(int(p, 16))[2:].zfill(nbits)
        except (ValueError, TypeError):
            return None

    df["Payload"] = df.apply(to_bits, axis=1)
    before = len(df)
    df = df[df["Payload"].notna()].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[CanditoExtractor] dropped {dropped} rows with unparseable payloads")
    return df


class CanditoExtractor(FeatureExtractor):
    """Caches a modified_dataset CSV as a CANdito-shaped, bit-encoded DataFrame."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.output_dir = os.path.join(self.features_path, "Candito")
        os.makedirs(self.output_dir, exist_ok=True)
        self.extract_features()

    def extract_features(self):
        prefix = self.file_name[:-4] if self.file_name.endswith(".csv") else self.file_name
        out_path = os.path.join(self.output_dir, prefix + "_raw.pkl")

        print(f"\n{'='*60}")
        print(f"  CanditoExtractor — Feature Extraction")
        print(f"{'='*60}")
        print(f"  Input          : {self.file_path}")

        df = self._load_raw_csv(self.file_path)
        df = _normalize_payload_column(df)

        df.to_pickle(out_path)
        n_attack = int(df["IsTampered"].sum())
        print(f"  Rows           : {len(df)} (benign: {len(df) - n_attack}, attack: {n_attack})")
        print(f"  CAN IDs        : {df['Id'].nunique()}")
        print(f"  Saved          : {out_path}")

    @staticmethod
    def _load_raw_csv(path):
        """Parse a `modified_dataset` CSV: timestamp,can_id,dlc,byte0..byte7,flag.

        Rows are always padded to 8 byte fields regardless of the row's own
        DLC (HCRL Car-Hacking convention, see
        datasets/CarHackingDataset/preprocess_dataset.py's CH_to_CANbusData:
        `data = (8 - dlc) * ["00"] + real_bytes`) -- the padding is on the
        LEFT, so for a short-DLC row the real payload is the LAST `dlc`
        byte fields, not the first (confirmed empirically: CAN ID 02b0's
        DLC=5 rows in Fuzzy_target.csv have a byte that changes every row
        only in the last of the 8 fields, with the first 3 constant zero).
        An optional header row ("timestamp,can_id,...") is skipped if present.
        """
        times, ids, dlcs, payloads, tampered = [], [], [], [], []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 4 or parts[0].strip().lower() == "timestamp":
                    continue
                flag = parts[-1].strip().upper()
                if flag not in ("R", "T"):
                    continue
                dlc = int(parts[2])
                byte_fields = parts[3:11]                    # always 8 padded fields
                real_bytes = byte_fields[8 - dlc:] if dlc else []
                payload_hex = "".join(b.strip().zfill(2) for b in real_bytes)
                times.append(float(parts[0]))
                ids.append(parts[1].strip())
                dlcs.append(dlc)
                payloads.append(payload_hex)
                tampered.append(1 if flag == "T" else 0)

        return pd.DataFrame({
            "Time": times, "Id": ids, "Dlc": dlcs,
            "Payload": payloads, "IsTampered": tampered,
        })
