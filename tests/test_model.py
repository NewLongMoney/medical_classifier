import sys
import os
import pytest
import torch
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import MedicalCNN

@pytest.fixture
def model_params():
    """Fixture for common model parameters"""
    return {
        'num_classes': 2,
        'batch_size': 32,
        'input_channels': 3,
        'input_height': 224,
        'input_width': 224
    }

@pytest.fixture
def model(model_params):
    """Fixture for model instance"""
    return MedicalCNN(num_classes=model_params['num_classes'])

def test_model_initialization(model, model_params):
    """Test if model initializes with correct parameters"""
    assert isinstance(model, nn.Module), "Model should be a PyTorch Module"
    assert model.fc2.out_features == model_params['num_classes'], "Output dimension should match num_classes"
    assert isinstance(model.conv1, nn.Conv2d), "First layer should be Conv2D"
    assert isinstance(model.conv2, nn.Conv2d), "Second layer should be Conv2D"

def test_model_forward_pass(model, model_params):
    """Test if forward pass works with correct dimensions"""
    batch_size = model_params['batch_size']
    input_tensor = torch.randn(
        batch_size, 
        model_params['input_channels'], 
        model_params['input_height'], 
        model_params['input_width']
    )
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    # Check output dimensions
    expected_shape = (batch_size, model_params['num_classes'])
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

def test_model_training_mode(model, model_params):
    """Test if model properly switches between train and eval modes"""
    # Test training mode
    model.train()
    assert model.training, "Model should be in training mode"
    
    # Test evaluation mode
    model.eval()
    assert not model.training, "Model should be in evaluation mode"

def test_model_parameter_gradients(model, model_params):
    """Test if gradients are properly computed during backward pass"""
    input_tensor = torch.randn(
        1, 
        model_params['input_channels'], 
        model_params['input_height'], 
        model_params['input_width']
    )
    
    # Forward pass
    model.train()
    output = model(input_tensor)
    
    # Create dummy target
    target = torch.tensor([0])
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} should not be None"

def test_model_output_range(model, model_params):
    """Test if model outputs are in the expected range after softmax"""
    input_tensor = torch.randn(
        1, 
        model_params['input_channels'], 
        model_params['input_height'], 
        model_params['input_width']
    )
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
    
    # Check if probabilities sum to 1 and are in [0,1]
    assert torch.allclose(probabilities.sum(dim=1), torch.tensor(1.0)), "Probabilities should sum to 1"
    assert (probabilities >= 0).all() and (probabilities <= 1).all(), "Probabilities should be in [0,1]"

def test_model_with_different_batch_sizes(model, model_params):
    """Test if model handles different batch sizes correctly"""
    batch_sizes = [1, 4, 16, 32]
    
    for batch_size in batch_sizes:
        input_tensor = torch.randn(
            batch_size, 
            model_params['input_channels'], 
            model_params['input_height'], 
            model_params['input_width']
        )
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, model_params['num_classes']), \
            f"Failed for batch size {batch_size}"

@pytest.mark.parametrize("num_classes", [2, 4, 8])
def test_model_with_different_num_classes(num_classes, model_params):
    """Test if model works with different numbers of output classes"""
    model = MedicalCNN(num_classes=num_classes)
    input_tensor = torch.randn(
        1, 
        model_params['input_channels'], 
        model_params['input_height'], 
        model_params['input_width']
    )
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output.shape == (1, num_classes), \
        f"Failed for num_classes={num_classes}"

def test_model_device_transfer(model, model_params):
    """Test if model can be moved between devices"""
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        assert next(model_cuda.parameters()).is_cuda, "Model should be on CUDA"
        
        model_cpu = model_cuda.cpu()
        assert not next(model_cpu.parameters()).is_cuda, "Model should be on CPU" 