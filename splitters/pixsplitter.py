# PixNet splitter — work in progress, will be released upon publication
from .base import BaseSplitter
import os
import re
import shutil
import pandas as pd


class PixSplitter(BaseSplitter):
    def __init__(self, input_dir, feature_extractor, cfg):
        super().__init__(input_dir)
        self.split_ratio       = cfg['split_ratio']
        self.feature_extractor = feature_extractor
        self.cfg               = cfg

    def split(self):
        # PixNet split logic — work in progress, will be released upon publication
        split_and_store_data(self.cfg)


def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(0)) if match else float('inf')


valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def extract_files(src_folder):
    return [
        f for f in os.listdir(src_folder)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]


def sequential_split_images(src_folder, train_folder, test_folder, split_ratio=0.2):
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder,  exist_ok=True)

    images      = extract_files(src_folder)
    total       = len(images)
    split_index = total - int(total * split_ratio)

    sorted_images = sorted(images, key=extract_number)
    train_images  = sorted_images[:split_index]
    test_images   = sorted_images[split_index:]

    for img in train_images:
        shutil.copy(os.path.join(src_folder, img), os.path.join(train_folder, img))
    for img in test_images:
        shutil.copy(os.path.join(src_folder, img), os.path.join(test_folder, img))

    return train_images, test_images


def split_labels(label_file, train_images, test_images, train_label_file, test_label_file):
    """Write each side's labels from `label_file`, the source of THIS call.

    A side with no images is skipped rather than written empty. run_all_ids.py
    expresses a two-file request as two passes over this function with
    split_ratio pinned to 0.0 (whole file -> train) and then 1.0 (whole file ->
    test). The second pass has no train images and must leave the train label
    file the first pass wrote alone; truncating it there is what silently
    replaced the train labels with the test file's, mislabelling 52% of the
    training set.
    """
    labels = {}
    with open(label_file, "r") as f:
        for line in f:
            if ":" in line:
                img, lab = line.strip().split(":", 1)
                labels[img.strip()] = lab.strip()

    for images, out_path in ((train_images, train_label_file),
                             (test_images, test_label_file)):
        if not images:
            continue
        written = 0
        with open(out_path, "w") as f:
            for img in images:
                if img in labels:
                    f.write(f"{img}: {labels[img]}\n")
                    written += 1
        missing = len(images) - written
        print(f"  labels -> {os.path.basename(os.path.dirname(out_path))}/"
              f"{os.path.basename(out_path)}: {written} written"
              + (f", {missing} image(s) had no label in the source" if missing else ""))


def split_track_csv(track_csv, train_images, test_images, train_csv, test_csv):
    df = pd.read_csv(track_csv)
    df.columns = df.columns.str.strip()

    train_img_nums = {int(''.join(filter(str.isdigit, img))) for img in train_images}
    test_img_nums  = {int(''.join(filter(str.isdigit, img))) for img in test_images}

    train_df = df[df["image_no"].isin(train_img_nums)]
    test_df  = df[df["image_no"].isin(test_img_nums)]

    # Same guard as split_labels: a side this call contributed no images to
    # keeps whatever the other pass wrote, rather than being truncated.
    if train_images:
        train_df.to_csv(train_csv, index=False)
    if test_images:
        test_df.to_csv(test_csv, index=False)

    print(f"Track split → Train rows: {len(train_df) if train_images else 'kept'}, "
          f"Test rows: {len(test_df) if test_images else 'kept'}")


def split_and_store_data(cfg):
    if not cfg['split']:
        return

    dir_path     = cfg['dir_path']
    dataset_name = cfg['dataset_name']
    file_name    = cfg['file_name']

    input_dir       = os.path.join(dir_path, "..", "datasets", dataset_name)
    train_dir       = os.path.join(input_dir, "train", cfg['train_dataset_dir'])
    test_dir        = os.path.join(input_dir, "test",  cfg['test_dataset_dir'])
    input_directory = os.path.join(input_dir, "features", "Images", file_name[:-4] + "_images")

    print("Splitting dataset into Train and Test")
    train_images, test_images = sequential_split_images(
        input_directory, train_dir, test_dir, cfg['split_ratio'])

    # PixNet label/track splitting — work in progress, will be released upon publication
    if cfg['feature_extractor'] == "PixNet":
        label_file       = os.path.join(input_directory, "labels.txt")
        train_label_file = os.path.join(train_dir, "labels.txt")
        test_label_file  = os.path.join(test_dir,  "labels.txt")
        split_labels(label_file, train_images, test_images, train_label_file, test_label_file)

        csv_file        = os.path.join(input_dir, "csv_files", file_name[:-4] + "_track.csv")
        train_track_csv = os.path.join(train_dir, "track.csv")
        test_track_csv  = os.path.join(test_dir,  "track.csv")
        split_track_csv(csv_file, train_images, test_images, train_track_csv, test_track_csv)
