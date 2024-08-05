import torch
from realesrgan import RealESRGANer
import os

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Check if the weight file exists
weight_file = 'weights/RealESRGAN_x2.pth'
if not os.path.exists(weight_file):
    print(f"Weights not found at '{weight_file}'. Please ensure the file is downloaded.")
else:
    print(f"Found weight file at '{weight_file}'.")

# Load the Real-ESRGAN model
try:
    print("Attempting to load Real-ESRGAN model...")
    model = RealESRGANer(device=device, scale=2, model_path=weight_file)
    if hasattr(model, 'model'):
        print("Model loaded, checking state_dict...")
        state_dict = model.model.state_dict()
        if state_dict is not None:
            print("Model state_dict loaded successfully.")
        else:
            print("Model state_dict is None.")
    else:
        print("Model object does not have 'model' attribute.")
except Exception as e:
    print(f"Error loading Real-ESRGAN model: {e}")
    model = None

# Verify model loaded correctly
if model is None or not hasattr(model, 'model') or model.model is None:
    print("Real-ESRGAN model could not be loaded. Exiting.")
else:
    print("Model initialized correctly.")
