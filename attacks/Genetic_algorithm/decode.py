"""
Decode perturbed genetic-algorithm frames back into real CAN traffic.

The 29x29x1 frame representation keeps only each packet's 29-bit arbitration ID —
timestamp, DLC, payload bytes and the R/T flag are all discarded by
GeneticExtractor. A perturbed frame therefore cannot be turned into a CSV on its
own; it has to be merged back into the rows it came from.

GeneticExtractor stores that provenance in its .npz (row_index, source_csv), so
this module rewrites *only* the arbitration ID of each affected row and copies
every other column verbatim from the source trace. The result is the original
traffic with the attacker-controlled IDs replaced by the ones the GA evolved —
which is exactly the perturbation the attack models — written in the standard
pipeline CSV format so any extractor or IDS in this repo can consume it.

Two output shapes:

  write_perturbed_trace  full-length trace: every source line preserved, so the
                         output has the same line count as the input and keeps
                         the surrounding traffic as temporal context. Only rows
                         belonging to perturbed frames have their ID rewritten.

  write_perturbed_csv    just the frames the attack evaluated (100 by default),
                         so the CSV corresponds one-to-one with the adversarial
                         .npz and the reported metrics. Frames are emitted in
                         source-trace order to keep timestamps monotonic.
"""

import csv
import os

import numpy as np


def bits_to_can_id(bits):
    """Convert a row of 29 frame bits (MSB first) to an integer CAN ID."""
    value = 0
    for b in bits:
        value = (value << 1) | (1 if b >= 0.5 else 0)

    return value


def _format_can_id(value, original):
    """
    Render a perturbed ID, keeping the trace's own hex width.

    Widens only when the evolved ID genuinely needs more nibbles than the
    original string had.
    """
    nibbles = (max(value, 1).bit_length() + 3) // 4
    width   = max(len(original.strip()), nibbles)

    return format(value, f"0{width}x")


def _id_overrides(frames, frame_indices, row_index):
    """
    Map source line number to the perturbed integer CAN ID for that line.

    Covers every row of every supplied frame; whether a row actually changed is
    decided later by comparing against the ID in the source trace, so untouched
    rows can be written back byte-for-byte.
    """
    frame_indices = np.asarray(frame_indices).astype(int)
    rows_per_frame = frames.shape[1]

    overrides = {}
    for pos, frame_idx in enumerate(frame_indices):
        for r in range(rows_per_frame):
            overrides[int(row_index[frame_idx][r])] = bits_to_can_id(frames[pos][r, :, 0])

    return overrides


def _check_source(source_csv):
    if not os.path.exists(source_csv):
        raise FileNotFoundError(
            f"Source trace not found: {source_csv}\n"
            f"It is recorded in the GeneticExtractor .npz — re-run Stage 1 if the "
            f"dataset moved."
        )


def _open_writer(path):
    """
    Open a CSV writer with LF terminators.

    csv.writer defaults to the excel dialect's CRLF; the traces in this repo are
    LF, and a stray \\r would ride along in the label column for any consumer
    that splits lines by hand instead of using csv.reader.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    handle = open(path, "w", newline="")

    return handle, csv.writer(handle, lineterminator="\n")


def _match_trailing_newline(source_csv, output_csv):
    """
    Mirror the source's trailing-newline convention.

    The CarHacking traces end without a final newline, so writing one would make
    `wc -l` report one more line than the input even though the content matches
    line for line.
    """
    def ends_with_newline(path, mode="rb"):
        with open(path, mode) as f:
            if f.seek(0, os.SEEK_END) == 0:
                return True          # empty file: nothing to trim
            f.seek(-1, os.SEEK_END)
            return f.read(1) == b"\n"

    if ends_with_newline(source_csv):
        return

    with open(output_csv, "rb+") as f:
        if f.seek(0, os.SEEK_END) == 0:
            return
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            f.truncate(f.tell() - 1)


def _write_labels(labels_csv, labels):
    handle, writer = _open_writer(labels_csv)
    with handle:
        writer.writerow(["frame_id", "label"])
        for i, label in enumerate(labels):
            writer.writerow([i, int(label)])


def write_perturbed_trace(frames, frame_indices, row_index, source_csv, output_csv,
                          labels_csv=None, labels=None):
    """
    Rewrite the full source trace with the perturbed IDs spliced in.

    Every line of the source is emitted — header, blank lines, unparseable rows
    and the tail packets that did not fill a complete frame included — so the
    output has exactly the same line count as the input and preserves the
    surrounding traffic as temporal context. Only rows belonging to a perturbed
    frame have their arbitration ID rewritten.

    Args:
        frames: (N, 29, 29, 1) perturbed frames
        frame_indices: (N,) index of each frame in the extractor's frame array
        row_index: (num_frames, 29) source line number per frame row
        source_csv: path to the raw CAN CSV the frames were built from
        output_csv: where to write the perturbed trace
        labels_csv: optional path for a frame_id,label sidecar
        labels: optional per-frame labels for the whole trace, for labels_csv

    Returns:
        dict with line/frame counts and how many IDs actually changed
    """
    _check_source(source_csv)

    overrides   = _id_overrides(frames, frame_indices, row_index)
    changed_ids = 0
    total_lines = 0

    handle, writer = _open_writer(output_csv)
    with handle, open(source_csv, "r") as src:
        for line_no, row in enumerate(csv.reader(src)):
            if line_no in overrides and row:
                original  = row[1]
                perturbed = overrides[line_no]
                if int(original, 16) != perturbed:
                    row = list(row)
                    row[1] = _format_can_id(perturbed, original)
                    changed_ids += 1

            writer.writerow(row)
            total_lines += 1

    _match_trailing_newline(source_csv, output_csv)

    if labels_csv and labels is not None:
        _write_labels(labels_csv, labels)

    return {
        'mode': 'full',
        'frames': len(np.asarray(frame_indices)),
        'rows': total_lines,
        'changed_ids': changed_ids,
        'output_csv': output_csv,
        'labels_csv': labels_csv if labels is not None else None,
    }


def write_perturbed_csv(frames, frame_indices, row_index, source_csv,
                        output_csv, labels_csv=None, labels=None):
    """
    Write only the frames the attack evaluated, as standalone CAN traffic.

    Frames are emitted in source-trace order (not the order the attack produced
    them) so timestamps stay monotonic — inter-arrival time matters to several of
    the IDSs in this repo, and the attack returns benign frames before attack
    frames regardless of where they sit in the trace.

    Args:
        frames: (N, 29, 29, 1) perturbed frames
        frame_indices: (N,) index of each frame in the extractor's frame array
        row_index: (num_frames, 29) source line number per frame row
        source_csv: path to the raw CAN CSV the frames were built from
        output_csv: where to write the perturbed traffic
        labels_csv: optional path for a frame_id,label sidecar
        labels: optional (N,) per-frame labels aligned with frames

    Returns:
        dict with row/frame counts and how many IDs actually changed
    """
    _check_source(source_csv)

    frame_indices = np.asarray(frame_indices).astype(int)

    # Pull just the rows these frames touch, in one pass over the trace.
    needed = set()
    for frame_idx in frame_indices:
        needed.update(int(ln) for ln in row_index[frame_idx])

    rows   = {}
    header = None
    with open(source_csv, "r") as f:
        for line_no, row in enumerate(csv.reader(f)):
            if line_no == 0:
                # Carry a header through verbatim so the output is structurally
                # identical to the input and whatever consumed one consumes the other.
                try:
                    int(row[1], 16)
                except (IndexError, ValueError):
                    header = row
            if line_no in needed:
                rows[line_no] = row

    missing = needed - rows.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} source rows referenced by row_index are missing from "
            f"{source_csv} — the trace changed since features were extracted. "
            f"Re-run Stage 1."
        )

    order          = np.argsort(frame_indices, kind='stable')
    changed_ids    = 0
    total_rows     = 0
    emitted_labels = []

    handle, writer = _open_writer(output_csv)
    with handle:
        if header is not None:
            writer.writerow(header)

        for pos in order:
            frame_idx = int(frame_indices[pos])

            for r in range(frames.shape[1]):
                source_row = list(rows[int(row_index[frame_idx][r])])
                original   = source_row[1]
                perturbed  = bits_to_can_id(frames[pos][r, :, 0])

                if int(original, 16) != perturbed:
                    source_row[1] = _format_can_id(perturbed, original)
                    changed_ids += 1

                writer.writerow(source_row)
                total_rows += 1

            if labels is not None:
                emitted_labels.append(int(labels[pos]))

    _match_trailing_newline(source_csv, output_csv)

    if labels_csv and labels is not None:
        _write_labels(labels_csv, emitted_labels)

    return {
        'mode': 'subset',
        'frames': len(frame_indices),
        'rows': total_rows,
        'changed_ids': changed_ids,
        'output_csv': output_csv,
        'labels_csv': labels_csv if labels is not None else None,
    }
