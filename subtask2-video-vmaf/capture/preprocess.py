import torch
import torchvision
from PIL import Image
import numpy as np
from realesrgan import RealESRGANer
import cv2
import os
import warnings

# Suppress warnings related to torchvision.transforms.functional_tensor
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")

# Print current working directory
print("Current Working Directory:", os.getcwd())

# Check versions of torch and torchvision
print("Torch version:", torch.__version__)

try:
    import torchvision.transforms.functional as F
    print("Torchvision version:", torchvision.__version__)
except ImportError:
    print("Torchvision is not installed. Please install it using 'pip install torchvision'.")
    exit()

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Check if the weight file exists
weight_file = 'weights/RealESRGAN_x2.pth'
if not os.path.exists(weight_file):
    print(f"Weights not found at '{weight_file}'. Please ensure the file is downloaded.")
    exit()

# Load the Real-ESRGAN model
try:
    model = RealESRGANer(device=device, scale=4, model_path=weight_file)
except AttributeError as e:
    print(f"Error loading Real-ESRGAN model: {e}")
    model = None

# Verify model loaded correctly
if model is None:
    print("Real-ESRGAN model could not be loaded. Exiting.")
    exit()

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Convert the frame to PIL Image for processing
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform super-resolution
    try:
        sr_image = model.predict(image)
    except Exception as e:
        print("Error during super-resolution:", e)
        break

    # Convert the output image to OpenCV format
    sr_image = cv2.cvtColor(np.array(sr_image), cv2.COLOR_RGB2BGR)

    # Resize super-resolved image back to the original size
    display_frame = cv2.resize(sr_image, (frame.shape[1], frame.shape[0]))

    # Display the enhanced video
    cv2.imshow('Enhanced Video (PyTorch - RealESRGAN)', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
