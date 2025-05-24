import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

@st.cache(allow_output_mutation=True)
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 2) 
    model.eval()
    return model

model = load_model()

# UI
st.title("Cats and Dogs Classifier")
st.write("Upload an image of a cat or dog and the model will predict it.")
st.write("Streamlit deployment for the cats and dogs classification activity")
st.write("Santiago A01383419")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_class = torch.argmax(probs).item()

    class_names = ["Cat", "Dog"]
    st.write(f"Prediction: {class_names[pred_class]}")
