import torch
import torch.nn as nn
from torchvision import models
import os

print("--- 🔍 VERIFYING MODEL CHECKPOINT LOADING ---")

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# 2. Recreate DenseNet121 Architecture
model = models.densenet121(weights=None)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)

# 3. Model Weight Loading
model_path = os.path.join("models", "densenet121_mixed_best.pth")

if not os.path.exists(model_path):
    print(f"❌ Error: Model file '{model_path}' nahi mili!")
else:
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("✅ SUCCESS: Model Checkpoint successfully loaded without any errors!")
    except Exception as e:
        print(f"❌ Error while loading checkpoint: {e}")