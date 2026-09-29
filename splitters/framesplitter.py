from .base import BaseSplitter
import os
import csv
import numpy as np


class FrameSplitter(BaseSplitter):
    def __init__(self, input_dir, feature_extractor, cfg):
        super().__init__(input_dir)
        self.split_ratio       = cfg['split_ratio']
        self.feature_extractor = feature_extractor
        self.cfg               = cfg

    def split(self):
        cfg       = self.cfg
        file_name = cfg['file_name']
        prefix    = file_name[:-4]

        features_path = self.feature_extractor.features_path
        frames_csv    = os.path.join(features_path, "Frames", prefix + "_frames.csv")
        labels_csv    = os.path.join(features_path, "Frames", prefix + "_labels.csv")
        print("Inside frame splitter")
        if not (os.path.exists(frames_csv) and os.path.exists(labels_csv)):
            print(f"  No frame features found at {os.path.dirname(frames_csv)}, skipping split.")
            return

        frames_all, labels_all = self.load_frames_and_labels(frames_csv, labels_csv)
        frames_all = np.array(frames_all)
        labels_all = np.array(labels_all)

        split_index = int((1 - self.split_ratio) * frames_all.shape[0])
        x_train = frames_all[:split_index]
        y_train = labels_all[:split_index]
        x_test  = frames_all[split_index:]
        y_test  = labels_all[split_index:]

        train_dir = os.path.join(self.input_dir, "train", cfg['train_dataset_dir'])
        test_dir  = os.path.join(self.input_dir, "test",  cfg['test_dataset_dir'])
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir,  exist_ok=True)

        train_frames_csv = os.path.join(train_dir, prefix + "_train_frames.csv")
        test_frames_csv  = os.path.join(test_dir,  prefix + "_test_frames.csv")

        if not os.path.exists(train_frames_csv):
            print("Creating trian frames")
            self.save_frames_and_labels(
                x_train, y_train,
                os.path.join(train_dir, prefix + "_train_frames.csv"),
                os.path.join(train_dir, prefix + "_train_labels.csv")
            )

        if not os.path.exists(test_frames_csv):
            print("Creating test frames")
            self.save_frames_and_labels(
                x_test, y_test,
                os.path.join(test_dir, prefix + "_test_frames.csv"),
                os.path.join(test_dir, prefix + "_test_labels.csv")
            )

        print(f"  Split          : Train={len(y_train)}, Test={len(y_test)}")

    def load_frames_and_labels(self, frames_csv, labels_csv):
        labels = []
        with open(labels_csv, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                labels.append(int(row[1]))

        num_frames = len(labels)

        frame_rows = []
        with open(frames_csv, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                frame_rows.append([int(x) for x in row])

        frame_rows = np.array(frame_rows)
        rows   = len(frame_rows) // num_frames
        bits   = frame_rows.shape[1]
        frames = frame_rows.reshape(num_frames, rows, bits, 1)

        return frames, np.array(labels)

    def save_frames_and_labels(self, frames, frame_labels, frames_csv, labels_csv):
        num_frames, rows, bits, _ = frames.shape

        with open(frames_csv, "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(num_frames):
                for r in range(rows):
                    writer.writerow(frames[i][r, :, 0].tolist())

        with open(labels_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_id", "label"])
            for i, label in enumerate(frame_labels):
                writer.writerow([i, label])
