import torch
from torchvision import transforms
from PIL import Image
from model import MedicalCNN
import config

def load_model(model_path, num_classes):
    model = MedicalCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    return model

def predict(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    
    return predicted.item()

if __name__ == "__main__":
    model_path = "models/best_model.pth"
    image_path = "path/to/your/image.jpg"
    num_classes = 2  # Update this based on your model
    device = torch.device(config.TRAIN_CONFIG["device"])
    
    model = load_model(model_path, num_classes).to(device)
    prediction = predict(image_path, model, device)
    print(f"Predicted class: {prediction}") 