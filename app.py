import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ===== CONFIG =====
MODEL_PATH = "image_classifier.pth"
TRAIN_DIR = r"D:\Fruits Classification\Training"  # For class names

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOAD CLASS NAMES =====
class_names = sorted(os.listdir(TRAIN_DIR))

# ===== MODEL LOADING =====
@st.cache_resource
def load_model():
    num_classes = len(class_names)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

# ===== IMAGE TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== STREAMLIT UI =====
st.set_page_config(page_title="🍎 Fruit Classifier", page_icon="🍇", layout="centered")

st.title("🍎 Fruit Classifier")
st.markdown("Upload a fruit image, and the AI will guess what it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    st.success(f"### 🏷 Prediction: **{predicted_class}**")
