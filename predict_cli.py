import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

print("--- 🏥 CHEST X-RAY HERNIA DETECTION (CLI INFERENCE) ---")

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Recreate Model Architecture
model = models.densenet121(weights=None)
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)

# 3. Load Checkpoint
model_path = os.path.join("models", "densenet121_mixed_best.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

classes = ['Hernia', 'Normal']

# 4. Preprocessing Exact Training Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_single_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ Error: File '{img_path}' nahi mili. Check path!")
        return

    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"❌ Invalid Image File: {e}")
        return

    # Preprocess & Predict
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        conf, predicted = torch.max(probs, 0)
        
    pred_class = classes[predicted.item()]
    confidence = conf.item() * 100
    
    print("\n" + "="*45)
    print(f" 🎯 PREDICTED CLASS : {pred_class.upper()}")
    print(f" 📊 CONFIDENCE      : {confidence:.2f}%")
    print(f" 📈 PROBABILITIES   -> Hernia: {probs[0].item()*100:.2f}% | Normal: {probs[1].item()*100:.2f}%")
    print("="*45 + "\n")

# 5. Interactive Loop
if __name__ == "__main__":
    while True:
        user_input = input("📸 Input Image Path daalein (ya 'exit' likhein band karne ke liye): ").strip()
        # Quotes remove agar drag-and-drop kiya ho
        user_input = user_input.replace('"', '').replace("'", "")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("👋 Exiting Inference CLI.")
            break
            
        if user_input:
            predict_single_image(user_input)