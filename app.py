# app.py

import streamlit as st
from PIL import Image
import os
import glob

st.set_page_config(layout="wide", page_title="Influencer Network Visualizations")

st.title("📊 Influencer Network Visualizations")

# Automatically find all PNG images in the visualizations directory
image_folder = "data/outputs/visualizations"
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))

if not image_paths:
    st.warning("No visualizations found. Please run the notebook to generate charts.")
else:
    for path in image_paths:
        title = os.path.splitext(os.path.basename(path))[0].replace("_", " ").capitalize()
        st.subheader(title)
        image = Image.open(path)
        st.image(image, use_column_width=True)

st.success("Visualizations rendered successfully.")
