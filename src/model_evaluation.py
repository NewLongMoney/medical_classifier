import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ModelEvaluator:
    """Evaluate model performance and generate visualizations."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate(self, data_loader, criterion):
        """Evaluate model on given data loader."""
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
