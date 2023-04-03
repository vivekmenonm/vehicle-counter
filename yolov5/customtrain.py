# Import necessary libraries
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import SGD
from torchvision import datasets, transforms

import yaml
from pathlib import Path
from tqdm import tqdm
from yolov5.models import YOLOv5
from yolov5.utils.datasets import LoadImagesAndLabels
from yolov5.utils.loss import ComputeLoss

# Load data.yaml
with open('data.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
lr = 0.001
batch_size = 16
epochs = 100

# Define transforms for the training data
transforms_train = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

# Load training data
dataset_train = LoadImagesAndLabels(data['train'], data['nc'], transforms=transforms_train)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=dataset_train.collate_fn)

# Define transforms for the validation data
transforms_val = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Load validation data
dataset_val = LoadImagesAndLabels(data['val'], data['nc'], transforms=transforms_val)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset_val.collate_fn)

# Define the model architecture
model = YOLOv5(num_classes=data['nc']).to(device)

# Define the loss function
criterion = ComputeLoss(model)

# Define the optimizer
optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

# Train the model
for epoch in range(epochs):
    # Train the model for one epoch
    model.train()
    train_loss = 0
    for batch_idx, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader_train)):
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        loss, _, _ = criterion(model(imgs), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Evaluate the model on the validation data
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader_val)):
            imgs = imgs.to(device)
            targets = targets.to(device)
            loss, _, _ = criterion(model(imgs), targets)
            val_loss += loss.item()

    # Print the epoch loss and validation loss
    train_loss /= len(dataloader_train)
    val_loss /= len(dataloader_val)
    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'custom_model.pt')
