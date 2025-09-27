import os
import shutil
import streamlit as st
from ultralytics import YOLO
from roboflow import Roboflow
from PIL import Image
import io

# --- Constants ---
MODEL_PATH = "best.pt"
ROBOFLOW_API_KEY_ENV = "ROBOFLOW_API_KEY"

# --- Model Training Function ---
def train_model():
    """
    Downloads data from Roboflow, trains the YOLOv8 model,
    and moves the best weights to the root directory.
    """
    # Initialize Roboflow
    api_key = os.environ.get(ROBOFLOW_API_KEY_ENV)
    if not api_key:
        return False, "Roboflow API key is not set in environment variables."
        
    try:
        rf = Roboflow(api_key=api_key)
        # Using a reliable public project for this example
        project = rf.workspace("augmented-library").project("waste-classification-aolds")
        dataset = project.version(1).download("yolov8")
        data_yaml_path = os.path.join(dataset.location, "data.yaml")
    except Exception as e:
        return False, f"Error connecting to Roboflow or downloading dataset: {e}"

    # Train the YOLOv8 model
    try:
        model = YOLO('yolov8n.pt')  # Load a pretrained model
        results = model.train(data=data_yaml_path, epochs=75, imgsz=640, project="YOLOv8-Waste-Training")
        
        # Find the path to the best model weights
        # The path is usually runs/detect/train/weights/best.pt
        source_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        
        if os.path.exists(source_path):
            shutil.move(source_path, MODEL_PATH)
            # Optional: Clean up the training directory
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
        try:
            return YOLO(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    model = load_model(MODEL_PATH)
    
    if model:
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
                result_image_bgr = results[0].plot()  # plot() returns a BGR numpy array
                result_image_rgb = result_image_bgr[..., ::-1]  # Convert BGR to RGB for PIL/Streamlit
            
            with col2:
                st.image(result_image_rgb, caption='Processed Image', use_column_width=True)

            detected_objects = {}
            for r in results:
                for c in r.boxes.cls:
                    class_name = model.names[int(c)]
                    detected_objects[class_name] = detected_objects.get(class_name, 0) + 1
            
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

    # If model doesn't exist, show the training interface
    if not os.path.exists(MODEL_PATH):
        st.warning(f"**Trained model (`{MODEL_PATH}`) not found.**")
        st.info("The model needs to be trained once. This process requires a Roboflow API key and may take some time.")
        
        st.markdown("""
        To proceed, please set the `ROBOFLOW_API_KEY` environment variable.
        If you don't have one, you can get it for free from [Roboflow](https://app.roboflow.com/).
        """)

        api_key = os.environ.get(ROBOFLOW_API_KEY_ENV)
        if not api_key:
            st.error("Your Roboflow API Key is not set. Please follow the instructions in Step 2 and restart the app.")
        else:
            st.success("Roboflow API key found in environment variables.")

        if st.button("🚀 Start Model Training", disabled=(not api_key)):
            with st.spinner("Training model... Check terminal for progress. This will take several minutes."):
                success, message = train_model()
            
            if success:
                st.success(message)
                st.success("Training complete! The app will now reload to use the new model.")
                st.balloons()
                st.experimental_rerun()  # Rerun the script to load the main app
            else:
                st.error(f"Training Failed: {message}")
    
    # If model exists, run the main application
    else:
        run_main_app()
