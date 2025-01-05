import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.model import MedicalCNN  # Update import path

def test_model_initialization():
    num_classes = 2
    model = MedicalCNN(num_classes)
    assert model.fc2.out_features == num_classes

def test_model_forward_pass():
    num_classes = 2
    model = MedicalCNN(num_classes)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (1, num_classes) 