import torch
import cv2
import os
import numpy as np
from models.RRDBNet_arch import RRDBNet

# Load pre-trained ESRGAN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RRDBNet(3, 3, 64, 23, gc=32)
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current script's directory
model_path = os.path.join(base_dir, "..", "models", "RRDB_ESRGAN_x4.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model = model.to(device)

def enhance_image(img):
    """Enhance the resolution of a license plate image using ESRGAN."""
    # Convert OpenCV image to Tensor
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # Normalize & add batch dim
    img = img.to(device)

    with torch.no_grad():
        output = model(img).clamp(0, 1)  # Forward pass through ESRGAN

    # Convert back to OpenCV format
    output = output.squeeze(0).cpu().numpy()
    output = np.transpose(output, (1, 2, 0))  # CHW to HWC
    output = (output * 255.0).astype(np.uint8)  # Denormalize
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)  # Convert back to BGR
