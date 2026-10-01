import csv
import os

import numpy as np

from features.feature_extractors.base import FeatureExtractor


class GeneticExtractor(FeatureExtractor):
    """
    Feature extractor dedicated to the genetic-algorithm evasion attacks
    (attacks/Genetic_algorithm/).

    Those attacks never read the Frames CSVs: GeneticAttack.__init__ calls
    np.load() on a single .npz and expects x_test / y_test, plus ecu_control for
    the spoof variant. This extractor writes that .npz directly, at the exact
    path attacks/attack_handler/genetic_attack.py rebuilds:

        datasets/<dataset_name>/test/<test_dataset_dir>/<file_name stem>_test_data.npz

    No train/test split is applied — the whole input CSV becomes the attack's
    evaluation set, so point file_name at a test trace and leave
    dataset_processing.split off.

    Frame geometry is ROWS x BITS x 1, set by dataset_processing.GeneticExtractor
    frame_rows / frame_bits — 29 x 29 for the Keras CANShield models (default),
    32 x 11 for a PyTorch MULSAM target.

    ecu_control is a (num_frames, ROWS) 0/1 matrix marking, per row of each frame,
    the CAN messages the attacker is assumed to control and may therefore
    rewrite. Selected via dataset_processing.GeneticExtractor.ecu_control_mode:

      id_owned   rows whose arbitration ID belongs to the compromised ECU
                 (ecu_ids) — the "transmitter controlled" reading used by
                 AdversarialSpoofAttack.find_dummy_rows
      zero_rows  rows whose ID is all zeros — the DoS-style free-slot reading.
                 Note FrameBuilder-style frames have no padding: every row is a
                 real message, so this only ever selects ID 0x000 traffic.
    """

    # ── ORIGINAL ─────────────────────────────────────────────────────────────
    # ROWS = 29
    # BITS = 29
    # ─────────────────────────────────────────────────────────────────────────
    # CHANGED: these are now DEFAULTS, overridable per dataset from config (see
    # __init__). Hardcoding them to the MULSAM geometry would silently break the
    # existing 29x29 Keras genetic attacks, which share this extractor.
    #   29 x 29  CANShield frames  — 29 packets x 29-bit extended arbitration ID
    #   32 x 11  MULSAM windows    — 32 packets x 11-bit standard ID
    ROWS = 29
    BITS = 29

    def __init__(self, cfg):
        super().__init__(cfg)

        sub = (cfg.get('dataset_processing') or {}).get('GeneticExtractor') or {}

        # ADDED: frame geometry from config. Instance attributes shadow the class
        # defaults above, so an unset config reproduces the original 29x29 output
        # byte for byte. Set frame_rows: 32 / frame_bits: 11 to target MULSAM.
        self.ROWS = int(sub.get('frame_rows', self.ROWS))
        self.BITS = int(sub.get('frame_bits', self.BITS))

        self.mode       = sub.get('ecu_control_mode', 'id_owned')
        self.max_frames = sub.get('max_frames')                       # None -> keep every frame
        self.out_dir    = sub.get('output_dir') or cfg.get('test_dataset_dir', '')
        self.ecu_ids    = {int(str(i), 16) for i in (sub.get('ecu_ids') or [])}

        if self.mode not in ('id_owned', 'zero_rows'):
            raise ValueError(
                f"ecu_control_mode must be 'id_owned' or 'zero_rows', got {self.mode!r}"
            )
        if self.mode == 'id_owned' and not self.ecu_ids:
            raise ValueError(
                "ecu_control_mode 'id_owned' requires "
                "dataset_processing.GeneticExtractor.ecu_ids — the hex CAN IDs owned by the "
                "compromised ECU, e.g. ['043f']. Without them every ecu_control entry would "
                "be 0 and the spoof GA would skip every attack frame."
            )

        self.test_root = os.path.join(self.dir_path, "..", "datasets", self.dataset_name, "test")
        self.extract_features(self.file_path)

    def extract_features(self, file_path):
        can_ids, bitstrings, row_labels, line_numbers = self.parse_traffic(file_path)
        frames, labels, ecu_control, row_index = self.build_frames(
            can_ids, bitstrings, row_labels, line_numbers
        )
        self.save_npz(frames, labels, ecu_control, row_index, file_path)

    def parse_traffic(self, csv_file):
        """
        Single pass over the raw CAN log.

        The per-row R/T flag and the per-row CAN ID must be collected in the SAME
        filtered stream as the bitstrings: rows whose ID fails to parse are
        dropped and do not consume a slot in a 29-row block (mirrors
        FrameBuilder.build_frames), so reading them in a separate pass would
        silently drift the frame/row alignment.

        line_numbers records each kept packet's 0-based line in the source file so
        the adversarial decoder can recover the columns the frame encoding drops
        (timestamp, DLC, payload, R/T flag) and rewrite only the ID.

        Returns:
            (can_ids, bitstrings, row_labels, line_numbers) — four parallel
            per-packet lists
        """
        can_ids, bitstrings, row_labels, line_numbers = [], [], [], []
        skipped = 0

        with open(csv_file, "r") as f:
            for line_no, row in enumerate(csv.reader(f)):
                if not row:
                    continue

                try:
                    can_id = int(row[1], 16)
                except (IndexError, ValueError):
                    skipped += 1
                    continue

                # ── ORIGINAL ─────────────────────────────────────────────────
                # bitstring = format(can_id, f"0{self.BITS}b")
                # if len(bitstring) != self.BITS:      # ID wider than 29 bits
                #     skipped += 1
                #     continue
                # ─────────────────────────────────────────────────────────────
                # CHANGED: mask instead of skip. At BITS=11 any ID above 0x7FF
                # formats to 12+ characters, so the length guard would DROP that
                # packet — and a dropped packet shifts every following window
                # boundary, so the attack and the target model would disagree
                # about which 32 messages make up a frame. MULSAM itself keeps
                # the low 11 bits (`(val >> i) & 1 for i in range(10, -1, -1)`,
                # ids/mulsam.py), so masking is what mirrors the target.
                #
                # At BITS=29 this is a no-op in practice: 29 bits is the widest
                # a CAN identifier can be, so nothing was ever masked away.
                can_id    = can_id & ((1 << self.BITS) - 1)
                bitstring = format(can_id, f"0{self.BITS}b")

                can_ids.append(can_id)
                bitstrings.append(bitstring)
                line_numbers.append(line_no)

                lbl = row[-1].strip().upper()
                row_labels.append(1 if lbl in ["T", "1", "ATTACK", "A"] else 0)

        print(f"  Input          : {os.path.basename(csv_file)}")
        print(f"  Packets        : {len(bitstrings)} ({skipped} unparseable rows skipped)")

        return can_ids, bitstrings, row_labels, line_numbers

    def build_frames(self, can_ids, bitstrings, row_labels, line_numbers):
        """Group packets into ROWSxBITSx1 frames and derive labels + ecu_control + row_index."""
        num_frames = len(bitstrings) // self.ROWS
        if self.max_frames:
            num_frames = min(num_frames, int(self.max_frames))
        if num_frames == 0:
            raise ValueError(
                f"{len(bitstrings)} parseable packets is not enough for a single "
                f"{self.ROWS}-row frame"
            )

        frames      = np.zeros((num_frames, self.ROWS, self.BITS, 1), dtype=np.float32)
        labels      = np.zeros(num_frames, dtype=np.int64)
        ecu_control = np.zeros((num_frames, self.ROWS), dtype=np.int64)
        row_index   = np.zeros((num_frames, self.ROWS), dtype=np.int64)

        for i in range(num_frames):
            lo, hi = i * self.ROWS, (i + 1) * self.ROWS

            frames[i, :, :, 0] = [[int(b) for b in s] for s in bitstrings[lo:hi]]
            labels[i]          = max(row_labels[lo:hi])
            row_index[i]       = line_numbers[lo:hi]

            if self.mode == 'id_owned':
                ecu_control[i] = [1 if cid in self.ecu_ids else 0 for cid in can_ids[lo:hi]]
            else:  # zero_rows
                ecu_control[i] = [1 if cid == 0 else 0 for cid in can_ids[lo:hi]]

        return frames, labels, ecu_control, row_index

    def save_npz(self, frames, labels, ecu_control, row_index, source_csv):
        out_dir = os.path.join(self.test_root, self.out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Stem must match attacks/attack_handler/genetic_attack.py, which builds
        # the path from cfg['file_name'][:-4].
        out_path = os.path.join(out_dir, self.file_name[:-4] + "_test_data.npz")

        # row_index + source_csv are the provenance the adversarial decoder needs to
        # turn perturbed frames back into real CAN traffic.
        np.savez(out_path,
                 x_test=frames,
                 y_test=labels,
                 ecu_control=ecu_control,
                 row_index=row_index,
                 source_csv=os.path.abspath(source_csv))

        attack_mask = labels == 1
        modifiable  = ecu_control[attack_mask].sum(axis=1)
        usable      = int((modifiable >= 1).sum())
        mean_rows   = float(modifiable.mean()) if modifiable.size else 0.0

        print(f"  Frames         : {len(labels)} "
              f"(Benign: {int((labels == 0).sum())}, Attack: {int(attack_mask.sum())})")
        print(f"  ecu_control    : mode={self.mode}, "
              f"mean modifiable rows per attack frame = {mean_rows:.2f}")
        print(f"  Usable         : {usable} attack frames have >= 1 modifiable row")

        if usable == 0:
            print("  WARNING        : no attack frame has a modifiable row — the spoof GA will "
                  "skip every frame. Check ecu_ids / ecu_control_mode.")

        print(f"  Saved          : {os.path.basename(out_path)} -> {out_dir}")
