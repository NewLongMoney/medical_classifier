import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from pathlib import Path
from sklearn.manifold import TSNE

class VisualizationTool:
    """Tools for visualizing model behavior and results."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_sample_predictions(self, model, data_loader, class_names, num_samples=8):
        """Plot sample predictions with their ground truth labels."""
        model.eval()
        images, labels = next(iter(data_loader))
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

        fig = plt.figure(figsize=(15, 8))
        for idx in range(min(num_samples, len(images))):
            ax = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])
            img = images[idx].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img)
            
            color = 'green' if predicted[idx] == labels[idx] else 'red'
            ax.set_title(f'Pred: {class_names[predicted[idx]]}\nTrue: {class_names[labels[idx]]}',
                        color=color)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_predictions.png')
        plt.close()
