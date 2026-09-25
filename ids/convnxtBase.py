import sys
import os

from sklearn.model_selection import KFold

from ids.base import IDS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'features')))
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import torch.nn.functional as F
 # Define transformations and dataset paths
data_transforms = {
    'test': transforms.Compose([transforms.ToTensor()]),
    'train': transforms.Compose([transforms.ToTensor()])
}

class ConvNeXtBase(IDS):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        # convnext_base.classifier is a Sequential; replace the final Linear layer
        # to match our binary classification (2 classes)
        # obtain in_features from the last module
        if isinstance(self.model.classifier, nn.Sequential):
            # assume last element is Linear
            last_linear = self.model.classifier[-1]
            in_feats = last_linear.in_features if hasattr(last_linear, 'in_features') else None
            if in_feats is None:
                raise ValueError("Unexpected classifier structure, no in_features found")
            self.model.classifier[-1] = nn.Linear(in_feats, 2)
        else:
            # fallback for any other structure
            self.model.classifier = nn.Linear(getattr(self.model.classifier, 'in_features', 0), 2)

    def train(self, train_dataset_dir, X_train=None, Y_train=None, cfg=None, **kwargs):
        cfg = cfg or {}

        # Load the test and train datasets from multiple folders
        train_label_file = os.path.join(train_dataset_dir, "labels.txt")
        train_loader = self.load_dataset(train_dataset_dir, train_label_file, is_train=True)
        print("train dataset dir : ", train_dataset_dir)
        print("Label file train : ", train_label_file)
        print("Loaded train dataset")

        epochs = cfg.get('epochs', 10)

        # Train the model
        model = self.train_model(train_loader, self.device, self.model, epochs)
        self.model = model    
 
    def test(self, X_test=None, Y_test=None, cfg=None, **kwargs):
        cfg = cfg or {}
        print("Entered model's testing method")
        dataset_path = os.path.join(cfg.get('dir_path', ''), "..", "datasets", cfg.get('dataset_name', ''))
        test_dataset_dir = os.path.join(dataset_path, "test", cfg.get('test_dataset_dir', ''))
        test_label_file = os.path.join(test_dataset_dir, "labels.txt")
        test_loader = self.load_dataset(test_dataset_dir, test_label_file,is_train=False)
        print("Loaded test dataset")

        all_preds, all_labels = self.test_model(test_loader, self.device, self.model)
        return all_preds, all_labels

    def save(self, path):
        self.model.eval()
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(path)
        print("Model saved.")
 

    def predict(self, X_test):
        super().predict(X_test)

    def load(self, path):
        self.model = torch.jit.load(path)
        self.model.to(self.device)

    def load_labels(self, label_file):
        """Load image labels from the label file."""
        labels = {}
        with open(label_file, 'r') as file:
            for line in file:
                # print("LINE : ", line)
                filename, label = line.strip().replace("'", "").replace('"', '').split(': ')
                # print("FILENAME : ", filename, "Label : ", label.strip().split()[1])

                labels[filename.strip()] = int(label.strip().split()[1])
        return labels
    
    def load_dataset(self, data_dir, label_file, is_train):
        """Load datasets and create DataLoader."""
        image_labels = self.load_labels(label_file)
        transform = data_transforms['train' if is_train else 'test']

        class ImageLabelDataset(Dataset):
            def __init__(self, data_dir, image_labels, transform):
                self.samples = [
                    (os.path.join(data_dir, filename), label)
                    for filename, label in image_labels.items()
                    if os.path.exists(os.path.join(data_dir, filename))
                ]
                self.transform = transform

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
                return image, label

        dataset = ImageLabelDataset(data_dir, image_labels, transform)
        batch_size = 32 if is_train else 1
        num_workers = 4 if is_train else 2
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=is_train,
            num_workers=num_workers, pin_memory=True, persistent_workers=True,
        )

        print(f'Loaded {len(dataset)} images.')
        return data_loader
    
    
    def train_model(self, train_loader, device, model , epochs):
    
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
        model = model.to(device)
    
        num_epochs = epochs
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
    
            model.train()
            running_loss = 0.0
            running_corrects = 0
    
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
    
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
    
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
    
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
        print("Training complete!")
    
        return model
    
    def k_fold_cross_validate(self,dataloader, device, model_type='convnext_base', k=5, batch_size=32):
        """
        Performs k-fold cross-validation given a DataLoader.
        It extracts the dataset from the dataloader and splits it.
        """
        dataset = dataloader.dataset  # Extract the original dataset
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        fold_accuracies = []

        best_acc = 0.0
        best_model_scripted = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"\n===== Fold {fold+1}/{k} =====")

            # Create subset datasets
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # Create DataLoaders from subsets
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # Train model on current fold
            model = self.train_model(train_loader, device, model_type)

            # Evaluate on validation set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            fold_accuracies.append(acc)
            print(f'Fold {fold+1} Accuracy: {acc:.2f}%')

            # Save best model
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'convnext_surrogate_gear.pth')
                print(f' Best model updated and saved (Acc: {best_acc:.2f}%)')

        print("\n=== Cross-validation complete ===")
        print(f'Average Accuracy: {np.mean(fold_accuracies):.2f}%')
        print(f'All Fold Accuracies: {fold_accuracies}')

    def test_model(self, test_loader, device,model):
    
        model.eval()
        # Initialize lists to store predictions and labels
        all_preds = []
        all_labels = []
    
        # Evaluate the model on the test dataset
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
            with torch.no_grad():  # Disable gradient calculation for evaluation
                outputs = model(inputs)  # Forward pass
                _, preds = torch.max(outputs, 1)  # Get predicted class labels
        
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
        # Convert lists to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
    
        # Calculate accuracy
        test_accuracy = np.sum(all_preds == all_labels) / len(all_labels)
        print(f'Test Accuracy: {test_accuracy:.4f}')
    
        return all_preds, all_labels  # Return accuracy for potential further use
