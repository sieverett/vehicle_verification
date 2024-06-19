import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import base64
from io import BytesIO

# Directory for the images
IMAGE_DIR = "vehicles"

# Create the directory if it does not exist
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def get_base64_image(image_path):
    img = Image.open(image_path)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# Function to create the dataframe
def create_image_dataframe(image_dir):
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    data = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        img_data = get_base64_image(image_path)
        data.append({
            'Filename': image_file,
            'Image': img_data,
            'Delete': False
        })
    return pd.DataFrame(data)

st.set_page_config(
        page_title="Vehicle Verification App",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

st.title("Vehicle Verification App")

# Image Manager in the sidebar
with st.sidebar.expander("Image Manager", expanded=True):
    uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(IMAGE_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved file: {uploaded_file.name}")

    df = create_image_dataframe(IMAGE_DIR)
    edited_df = st.data_editor(
        df,
        column_config={
            'Filename': 'Filename',
            'Image': st.column_config.ImageColumn("Image", width="small"),
            'Delete': st.column_config.CheckboxColumn(help='Check to delete this image')
        },
        disabled=['Filename', 'Image'],
        hide_index=True
    )

    if st.button('Delete Selected Images'):
        for index, row in edited_df.iterrows():
            if row['Delete']:
                file_path = os.path.join(IMAGE_DIR, row['Filename'])
                if os.path.exists(file_path):
                    os.remove(file_path)
                    st.warning(f"Deleted file: {row['Filename']}")
        st.rerun()  # Refresh the app to update the file list

# Vehicle verification code

# Load the pretrained ResNet50 model
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
model.eval()  # Set the model to evaluation mode

# Define a transformation to preprocess the images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalization values for ImageNet
])

def extract_features(image):
    image = image.convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()  # Remove batch dimension and convert to numpy array

def calculate_similarity(features1, features2):
    return cosine_similarity([features1], [features2])[0][0]

PICKLE_FILE = 'allowed_vehicles.pkl'

def save_allowed_vehicles(allowed_vehicles):
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(allowed_vehicles, f)

def load_allowed_vehicles():
    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def update_allowed_vehicles(directory):
    allowed_vehicles = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            features = extract_features(image)
            allowed_vehicles[filename] = features
    save_allowed_vehicles(allowed_vehicles)
    print("Allowed vehicles list updated.")

def is_vehicle_allowed(image, allowed_vehicles, threshold, return_table=False):
    features = extract_features(image)
    similarities = []

    for filename, allowed_features in allowed_vehicles.items():
        similarity = calculate_similarity(features, allowed_features)
        similarities.append((filename, similarity, similarity >= threshold))

    if return_table:
        df = pd.DataFrame(similarities, columns=['Filename', 'Similarity', 'Exceeds Threshold'])
        df['Image'] = df['Filename'].apply(lambda x: get_base64_image(f"vehicles/{x}"))
        df['Exceeds Threshold'] = df['Exceeds Threshold'].apply(lambda x: 'Yes' if x else 'No')
        df = df.sort_values(by='Similarity', ascending=False)
        return df

    for filename, similarity, exceeds in similarities:
        if exceeds:
            return True, filename

    return False, None

# Streamlit app
st.write("Upload an image of the vehicle to check if it is allowed.")

# Slider in the sidebar
with st.sidebar.expander("Settings", expanded=True):
    threshold = st.slider("Set the similarity threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.01)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to the current directory
    with open("vehicle.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Vehicle Image", width=200)
    
    # Update allowed vehicles list if there are any changes
    allowed_vehicles_directory = 'vehicles'
    update_allowed_vehicles(allowed_vehicles_directory)

    # Load allowed vehicles from the pickle file
    allowed_vehicles = load_allowed_vehicles()
    
    # Check if the vehicle is allowed
    new_vehicle_image_path = 'vehicle.png'
    new_image = Image.open(new_vehicle_image_path)
    allowed, matched_filename = is_vehicle_allowed(new_image, allowed_vehicles, threshold)
    if allowed:
        st.write(":green[Vehicle is allowed]")
    else:
        st.write("Vehicle is not allowed.")
    
    # Generate and display the DataFrame
    df = is_vehicle_allowed(new_image, allowed_vehicles, threshold, return_table=True)
    if df is not None:
        with st.expander("Similarity Table", expanded=False):
            st.dataframe(
                df,
                column_config={
                    "Filename": st.column_config.TextColumn("Filename"),
                    "Similarity": st.column_config.NumberColumn("Similarity"),
                    "Exceeds Threshold": st.column_config.TextColumn("Exceeds Threshold"),
                    "Image": st.column_config.ImageColumn("Image", width="small"),
                },
                use_container_width=True,
                hide_index=True,
            )
