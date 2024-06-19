import streamlit as st
import os
from PIL import Image
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

st.title("Vehicle Image Manager")

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

# Create the dataframe and display it
df = create_image_dataframe(IMAGE_DIR)

with st.sidebar.expander("Image Manager", expanded=True):
    uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(IMAGE_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved file: {uploaded_file.name}")

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
        st.experimental_rerun()  # Refresh the app to update the file list