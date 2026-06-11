
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class RandomSatelliteDataset(Dataset):
    """Generate synthetic satellite-like images for testing"""
    
    def __init__(self, num_samples=5000, image_size=64, num_classes=10, augmentation=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.augmentation = augmentation
        
        if augmentation is None:
            self.augmentation = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random synthetic image
        image = Image.new('RGB', (self.image_size, self.image_size), 
                         color=(np.random.randint(0, 255), 
                               np.random.randint(0, 255), 
                               np.random.randint(0, 255)))
        
        # Apply augmentations to create two views
        view1 = self.augmentation(image)
        view2 = self.augmentation(image)
        
        label = np.random.randint(0, self.num_classes)
        
        return view1, view2, label

def get_dataloader(batch_size=128, num_samples=5000, shuffle=True, num_workers=0):
    """Create DataLoader"""
    dataset = RandomSatelliteDataset(num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                     num_workers=num_workers, drop_last=True)
