TRAIN_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_dir": "./models"
} 