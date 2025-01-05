import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path

class DataPreprocessor:
    """Handle data preprocessing for medical images."""
    
    def __init__(self, input_size=(224, 224)):
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path, augment=False):
        """Preprocess a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            if augment:
                return self.augmentation(image)
            return self.transform(image)
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None 