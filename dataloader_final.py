import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted([cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # instantiate once here
        
        for cls in self.classes:
            cls_folder = os.path.join(data_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'tif','pgm')):
                    self.image_paths.append(os.path.join(cls_folder, fname))
                    self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            #load image in grayscale for CLAHE
            image = Image.open(img_path).convert("L")
            img_np = np.array(image)
            
            # Apply CLAHE for contrast enhancement
            img_clahe = self.clahe.apply(img_np)
            
            #convert to PIL Image and then to RGB (3 channels)
            img_clahe_rgb = Image.fromarray(img_clahe).convert("RGB")
            
            if self.transform:
                img_clahe_rgb = self.transform(img_clahe_rgb)
                
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}. Error: {e}")
            img_clahe_rgb = torch.zeros(3, 224, 224)  
            label = 0
            
        return img_clahe_rgb, label, img_path

def get_dataloaders(data_dir, batch_size, img_size, num_workers=4):
    print(f"Loading datasets from: {data_dir}")
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        
        transforms.RandomHorizontalFlip(p=0.5),  
        transforms.RandomRotation(degrees=(-5, 5), fill=0),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05), 
            scale=(0.95, 1.05), 
            fill=0
        ),
        
        transforms.ColorJitter(
            brightness=0.1,    
            contrast=0.1,    
            saturation=0,    
            hue=0           
        ),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    train_dataset = CustomImageDataset(train_dir, transform=train_transform)
    val_dataset = CustomImageDataset(val_dir, transform=val_test_transform)
    test_dataset = CustomImageDataset(test_dir, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }
    
    print("Dataset sizes:")
    print(f"  Train: {dataset_sizes['train']} images")
    print(f"  Val:   {dataset_sizes['val']} images")
    print(f"  Test:  {dataset_sizes['test']} images")
    print(f"  Classes: {train_dataset.classes}")
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    return dataloaders, dataset_sizes, train_dataset.classes
