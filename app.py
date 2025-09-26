import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from roboflow import Roboflow
import os
import zipfile
import shutil
from collections import Counter

# --- Page Configuration ---
st.set_page_config(
    page_title="Computer Vision-based Waste Segregation System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Roboflow Model ---
# Caching the model loading to prevent reloading on every interaction.
@st.cache_resource
def load_roboflow_model(api_key, model_id):
    """Loads and returns the Roboflow model object."""
    rf = Roboflow(api_key=api_key)
    project_id, version_number = model_id.split('/')
    project = rf.workspace().project(project_id)
    return project.version(int(version_number)).model

# --- Helper Functions ---
def plot_waste_distribution(predictions):
    """Generates a bar chart from the prediction counts."""
    class_counts = Counter(p['class'] for p in predictions)
    
    if not class_counts:
        return None

    df = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Count']).reset_index()
    df = df.rename(columns={'index': 'Waste Type'})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['Waste Type'], df['Count'], color='dodgerblue')
    ax.set_ylabel('Count')
    ax.set_title('Waste Segregation Analysis', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    
    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center') 
        
    return fig

# --- Main Application ---
st.title("♻️ Computer Vision Waste Segregation System for Smart Cities")
st.write("Leveraging YOLOv8 to automate waste detection and classification from images. Choose between analyzing a single image or processing a batch of images from a `.zip` file.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter your Roboflow API Key:", type="password")
    
    st.subheader("🤖 Model Details")
    model_id = st.text_input("Roboflow Model ID:", "waste-detection-p52ar/1")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)

# --- Tabbed Interface ---
tab1, tab2 = st.tabs(["🖼️ Single Image Analysis", "🗂️ Batch Dataset Analysis"])

# --- SINGLE IMAGE TAB ---
with tab1:
    st.header("Upload an Image for Analysis")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

      
