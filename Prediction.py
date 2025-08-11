import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --------- CONFIG ---------
MODEL_PATH = "image_classifier.pth"  # Your saved model from Step 2
CLASS_NAMES = sorted(os.listdir("Training"))  # Reads folder names from your training folder
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- LOAD MODEL ---------
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --------- IMAGE TRANSFORMS ---------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(img_path):
    # Load and preprocess
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_class = torch.max(probs, dim=1)

    return CLASS_NAMES[pred_class.item()], conf.item()

# --------- TEST PREDICTION ---------
if __name__ == "__main__":
    test_image = "D:\Fruits Classification\cucumber.jpg"  # Change to an actual test image path
    pred_label, confidence = predict_image(test_image)
    print(f"Predicted: {pred_label} | Confidence: {confidence:.2f}")
