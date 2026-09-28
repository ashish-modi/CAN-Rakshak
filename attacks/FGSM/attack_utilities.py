import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from .model import InceptionResNetV1

data_transforms = {
    'test': transforms.Compose([transforms.ToTensor()]),
    'train': transforms.Compose([transforms.ToTensor()]),
}


def stuff_bits(binary_string):
    """Insert '1' after every 5 consecutive '0's in the binary string."""
    result = ''
    count = 0
    for bit in binary_string:
        result += bit
        if bit == '0':
            count += 1
            if count == 5:
                result += '1'
                count = 0
        else:
            count = 0
    return result


def calculate_crc(data):
    """Calculate CRC-15 checksum for the given bit string."""
    crc = 0x0000
    poly = 0x4599
    for bit in data:
        crc ^= (int(bit) & 0x01) << 14
        for _ in range(15):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
        crc &= 0x7FFF
    return crc


def hex_id_to_bits(hex_id, bit_len=11):
    """Convert a hex CAN arbitration ID (e.g. '43f') into an 11-bit binary string."""
    return format(int(hex_id, 16), '0{}b'.format(bit_len))[-bit_len:]


def load_model(surrogate_model_path, target_model_path, device):
    """Load surrogate (gradient source) and target (evaluation) checkpoints as InceptionResNetV1 state dicts."""
    model = InceptionResNetV1(num_classes=2)
    test_model = InceptionResNetV1(num_classes=2)

    model.load_state_dict(torch.load(surrogate_model_path, map_location=device, weights_only=True))
    test_model.load_state_dict(torch.load(target_model_path, map_location=device, weights_only=True))

    model = model.to(device)
    test_model = test_model.to(device)

    model.eval()
    test_model.eval()

    return model, test_model


def load_labels(label_file):
    """
    Parse 'filename: v1, v2, ..., label' lines (the format used by every
    labels.txt in datasets/**/train|test/**) — the label is always the
    last comma-separated value.
    """
    labels = {}
    with open(label_file, 'r') as file:
        for line in file:
            filename, label_str = line.strip().replace("'", "").replace('"', '').split(': ')
            labels[filename.strip()] = int(label_str.strip().split(',')[-1].strip())
    return labels


def load_dataset(data_dir, label_file, is_train=True):
    image_labels = load_labels(label_file)

    images = []
    labels = []

    for filename, label in image_labels.items():
        img_path = os.path.join(data_dir, filename)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            transform = data_transforms['train'] if is_train else data_transforms['test']
            images.append(transform(image))
            labels.append(label)

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(images_tensor, labels_tensor)
    batch_size = 32 if is_train else 1
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'Loaded {len(images)} images.')

    return dataset, data_loader
