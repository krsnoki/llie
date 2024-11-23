import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import Restormer 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(checkpoint_path):
   
    model = Restormer(
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        channels=[48, 96, 192, 384],
        num_refinement=4,
        expansion_factor=2.66
    ).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    return model

def process_image(model, input_image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    input_tensor = transform(input_image).unsqueeze(0).to(device) 

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
    
    return output_image

# Streamlit app layout
st.title("Low Light Image Enhancement")
st.header("*Mini Project*")
st.subheader("Kalyani Prashant Kolte - 2022BIT503")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Uploaded Image", use_column_width=True)

        checkpoint_path = 'result/inpaint.pth'
        model = load_model(checkpoint_path)

        if model is not None:
           
            output_image = process_image(model, input_image)
            st.image(output_image, caption="Output Image", use_column_width=True)
        else:
            st.error("Failed to load the model.")
    
    except Exception as e:
        st.error(f"Error processing image: {e}")
