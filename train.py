import os
import argparse
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from models import get_model  
from dataloader import get_dataloaders  

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_and_save(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Loss plot
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epochs')
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epochs')
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()

def save_confusion_matrix(labels, preds, classes, save_dir):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    plt.figure(figsize=(8,8))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading datasets...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    num_classes = len(classes)
    print(f"Classes found: {classes}")

    print("Initializing model...")
    model = get_model(name=args.model_name, num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Optional learning rate warmup (simple linear warmup)
    if args.lr_warmup:
        warmup_steps = args.lr_warmup_steps
        def lr_lambda(current_step):
            if current_step >= warmup_steps:
                return 1
            return float(current_step) / float(max(1, warmup_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = eval_one_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if scheduler:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print("Training complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.model_name}_best.pth'))

    plot_and_save(history, args.save_dir)
    save_confusion_matrix(val_labels, val_preds, classes, args.save_dir)

    print(f"Model and plots saved in {args.save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train mammogram classification model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory with train/val/test folders')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save model and plots (configs folder)')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Model architecture name (resnet50, efficientnet_b4, etc.)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr_warmup', action='store_true', help='Enable learning rate warmup')
    parser.add_argument('--lr_warmup_steps', type=int, default=500, help='Number of warmup steps')
    parser.add_argument('--use_crop', action='store_true', help='Crop breast region from images before training')  # <-- ADD THIS
    args = parser.parse_args()

    main(args)

