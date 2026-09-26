"""
Shared, mode-independent helpers used by both dos.py and spoof.py:
model/dataset loading, CRC-15, and image saving. Kept identical between
the two attack modes -- anything that diverges (mask generation, queue
budgets, gradient perturbation, evaluation metrics) lives in dos.py /
spoof.py instead.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
from PIL import Image
from torchvision.utils import save_image


def load_model(model_path):
    num_classes = 2

    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    return model


data_transforms = {
    'test': transforms.Compose([transforms.ToTensor()]),
    'train': transforms.Compose([transforms.ToTensor()])
}


def load_labels(label_file):
    """Load image labels from the label file."""
    labels = {}
    with open(label_file, 'r') as file:
        for line in file:
            filename, label_str = line.strip().replace("'", "").replace('"', '').split(': ')
            label = int(label_str.strip().split(',')[-1].strip())
            labels[filename.strip()] = label
    return labels


def load_dataset(data_dir, label_file, device, is_train=True):
    image_labels = load_labels(label_file)

    images = []
    labels = []
    start_image_number = None

    for filename, label in image_labels.items():
        img_path = os.path.join(data_dir, filename)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            if is_train:
                image = data_transforms['train'](image)
            else:
                image = data_transforms['test'](image)
            images.append(image)
            labels.append(label)

            if start_image_number is None:
                start_image_number = int(filename.split('_')[-1].split('.')[0])

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(images_tensor, labels_tensor)
    batch_size = 32 if is_train else 1
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'Loaded {len(images)} images.')

    return dataset, data_loader, start_image_number


def calculate_crc(data):
    """Calculate CRC-15 checksum for the given data."""
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


def saving_image(img, name, output_path):
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f'perturbed_image_{name}.png')
    save_image(img, output_path)
