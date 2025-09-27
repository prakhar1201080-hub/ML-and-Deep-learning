# Save this as waste_sorter_app.py

import os
import sys
import shutil
import streamlit as st
from PIL import Image
import io

# This function is what is currently saving you from a bigger crash.
# It checks if you are in the correct environment.
def system_check():
    """Checks for a compatible Python version."""
    if sys.version_info >= (3, 12):
        st.error(
            f"❌ Incompatible Python Version (v{sys.version_info.major}.{sys.version_info.minor}) Detected\n\n"
            "This application's dependencies (like PyTorch and OpenCV) are not yet compatible with Python 3.12+.\n\n"
            "**Please create and activate a new virtual environment using Python 3.10 or 3.11 before running the app.**"
        )
        st.stop() # Stops the script to prevent crashing.

# --- Run the safety check first ---
system_check()

# --- If the check passes, these imports will now succeed ---
from ultralytics import YOLO
from roboflow import Roboflow

# (The rest of the application code remains the same)
# --- Constants ---
MODEL_PATH = "best.pt"
ROBOFLOW_API_KEY_ENV = "ROBOFLOW_API_KEY"

# --- Model Training Function ---
def train_model():
    api_key = os.environ.get(ROBOFLOW_API_KEY_ENV)
    if not api_key: return False, "Roboflow API key not set."
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("augmented-library").project("waste-classification-aolds")
        dataset = project.version(1).download("yolov8")
        model = YOLO('yolov8n.pt')
        results = model.train(data=os.path.join(dataset.location, "data.yaml"), epochs=75, imgsz=640, project="YOLOv8-Waste-Training")
        source_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        if os.path.exists(source_path):
            shutil.move(source_path, MODEL_PATH)
            shutil.rmtree("YOLOv8-Waste-Training")
            return True, f"Model trained and saved as `{MODEL_PATH}`"
        return False, "Could not find trained model file."
    except Exception as e:
        return False, f"Training error: {e}"

# --- Streamlit Main Application ---
def run_main_app():
    st.success(f"**Trained Model (`{MODEL_PATH}`) Loaded!**")
    st.write("Upload an image to detect waste.")
    @st.cache_resource
    def load_model(model_path): return YOLO(model_path)
    model = load_model(MODEL_PATH)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        col1, col2 = st.columns(2)
        col1.image(image, caption='Uploaded Image', use_column_width=True)
        with st.spinner("Processing..."):
            results = model(image)
            result_image = results[0].plot()[..., ::-1] # BGR to RGB
        col2.image(result_image, caption='Processed Image', use_column_width=True)
        detected_objects = {}
        for r in results:
            for c in r.boxes.cls:
                name = model.names[int(c)]
                detected_objects[name] = detected_objects.get(name, 0) + 1
        if detected_objects:
            st.write("### Detected Waste Types:")
            st.markdown("\n".join([f"- **{v}** `{k.capitalize()}`" for k, v in detected_objects.items()]))
        else:
            st.info("No waste detected.")

# --- Main Controller ---
if __name__ == "__main__":
    st.set_page_config(page_title="Smart Waste Segregation", layout="wide")
    st.title("♻️ AI-Powered Waste Segregation System")
    if not os.path.exists(MODEL_PATH):
        st.warning(f"**Trained model not found.** Please train the model.")
        if not os.environ.get(ROBOFLOW_API_KEY_ENV):
            st.error(f"Set `{ROBOFLOW_API_KEY_ENV}` environment variable.")
        else:
            st.success("Roboflow API key found.")
        if st.button("🚀 Start Model Training", disabled=(not os.environ.get(ROBOFLOW_API_KEY_ENV))):
            with st.spinner("Training... See terminal for progress."):
                success, msg = train_model()
            if success: st.success(msg); st.balloons(); st.experimental_rerun()
            else: st.error(f"Failed: {msg}")
    else:
        run_main_app()
