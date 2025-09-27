import os
import sys
import shutil
import streamlit as st
from PIL import Image
import io

# --- Pre-flight System and Python Version Check ---
def system_check():
    """
    Checks if the system is running a compatible Python version.
    Halts the app with an informative error if the version is too new.
    """
    if sys.version_info >= (3, 12):
        st.error(
            f"❌ Incompatible Python Version (v{sys.version_info.major}.{sys.version_info.minor}) Detected\n\n"
            "This application's dependencies (like PyTorch and OpenCV) are not yet compatible with Python 3.12+.\n\n"
            "**Please create a new virtual environment with Python 3.10 or 3.11.**\n"
            "Example commands:\n"
            "```bash\n"
            "# 1. Create the environment\n"
            "python3.10 -m venv venv\n\n"
            "# 2. Activate it\n"
            "source venv/bin/activate\n\n"
            "# 3. Re-install packages\n"
            "pip install -r requirements.txt\n"
            "```"
        )
        st.stop()  # Stop the Streamlit script execution

# --- Run the check before importing heavy libraries ---
system_check()

# --- Now, safely import the libraries that would have crashed ---
from ultralytics import YOLO
from roboflow import Roboflow

# --- Constants ---
MODEL_PATH = "best.pt"
ROBOFLOW_API_KEY_ENV = "ROBOFLOW_API_KEY"

# (The rest of the code is identical to the previous version)

# --- Model Training Function ---
def train_model():
    """
    Downloads data from Roboflow, trains the YOLOv8 model,
    and moves the best weights to the root directory.
    """
    api_key = os.environ.get(ROBOFLOW_API_KEY_ENV)
    if not api_key:
        return False, "Roboflow API key is not set in environment variables."
        
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("augmented-library").project("waste-classification-aolds")
        dataset = project.version(1).download("yolov8")
        data_yaml_path = os.path.join(dataset.location, "data.yaml")
    except Exception as e:
        return False, f"Error connecting to Roboflow or downloading dataset: {e}"

    try:
        model = YOLO('yolov8n.pt')
        results = model.train(data=data_yaml_path, epochs=75, imgsz=640, project="YOLOv8-Waste-Training")
        source_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        
        if os.path.exists(source_path):
            shutil.move(source_path, MODEL_PATH)
            shutil.rmtree("YOLOv8-Waste-Training")
            return True, f"Model trained and saved as `{MODEL_PATH}`"
        else:
            return False, "Could not find the trained model file 'best.pt' after training."
    except Exception as e:
        return False, f"An error occurred during model training: {e}"


# --- Streamlit Main Application ---
def run_main_app():
    """
    The main Streamlit application for uploading images and getting predictions.
    """
    st.success(f"**Trained Model (`{MODEL_PATH}`) Loaded Successfully!**")
    st.write("Upload an image to detect and classify different types of waste.")

    @st.cache_resource
    def load_model(model_path):
        """Cached function to load the YOLO model."""
        return YOLO(model_path)

    model = load_model(MODEL_PATH)
    
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with st.spinner("Processing image..."):
            results = model(image)
            result_image_bgr = results[0].plot()
            result_image_rgb = result_image_bgr[..., ::-1]
        
        with col2:
            st.image(result_image_rgb, caption='Processed Image', use_column_width=True)

        detected_objects = {model.names[int(c)]: 0 for r in results for c in r.boxes.cls}
        for r in results:
            for c in r.boxes.cls:
                class_name = model.names[int(c)]
                detected_objects[class_name] += 1
        
        if detected_objects:
            st.write("### Detected Waste Types:")
            items = [f"- **{count}** `{obj.capitalize()}`" for obj, count in detected_objects.items()]
            st.markdown("\n".join(items))
        else:
            st.info("No waste objects were detected.")

# --- Main Controller ---
if __name__ == "__main__":
    st.set_page_config(page_title="Smart Waste Segregation", layout="wide")
    st.title("♻️ AI-Powered Waste Segregation System")

    if not os.path.exists(MODEL_PATH):
        st.warning(f"**Trained model (`{MODEL_PATH}`) not found.**")
        st.info("The model needs to be trained once. This process requires a Roboflow API key and may take some time.")
        
        if not os.environ.get(ROBOFLOW_API_KEY_ENV):
            st.error(f"Please set the `{ROBOFLOW_API_KEY_ENV}` environment variable and restart the app.")
        else:
            st.success("Roboflow API key found.")

        if st.button("🚀 Start Model Training", disabled=(not os.environ.get(ROBOFLOW_API_KEY_ENV))):
            with st.spinner("Training model... Check terminal for progress. This may take several minutes."):
                success, message = train_model()
            
            if success:
                st.success(message)
                st.balloons()
                st.experimental_rerun()
            else:
                st.error(f"Training Failed: {message}")
    else:
        run_main_app()
