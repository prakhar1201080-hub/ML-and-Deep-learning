import streamlit as st
from PIL import Image
from roboflow import Roboflow
import matplotlib.pyplot as plt
import pandas as pd
import os
import requests

# --- Helper Functions ---
def plot_results_on_image(image, results):
    """
    Plots bounding boxes on an image. This is a simplified version;
    for more advanced plotting, consider libraries like OpenCV.
    """
    # For simplicity, we will just return the image for now.
    # Advanced plotting requires more complex code (e.g., with OpenCV).
    # Streamlit will display the annotated image from Roboflow's prediction.
    return image

# --- Streamlit App ---

st.set_page_config(layout="wide")

st.title("️♻️ Computer Vision-based Waste Segregation System")
st.write("""
    Upload an image of waste items, and the YOLOv8 model will detect and classify them. 
    This system is designed for smart cities to automate waste management.
    """)

# --- Sidebar for User Inputs ---

with st.sidebar:
    st.header("Configuration")
    # Get API Key from the user
    api_key = st.text_input("Enter your Roboflow API Key:", type="password")
    
    # Model Details
    st.subheader("YOLOv8 Model Details")
    model_id = st.text_input("Roboflow Model ID:", "waste-detection-p52ar/1")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Image Upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Example Images
    st.subheader("Or use an example image:")
    example_images = {
        "Example 1": "https://storage.googleapis.com/com-roboflow-cms-staging/760_trash_in_nature.jpg",
        "Example 2": "https://storage.googleapis.com/com-roboflow-cms-staging/recycling_bins.jpg",
    }
    selected_example = st.selectbox("Select an example:", list(example_images.keys()))

    # Button to start detection
    run_button = st.button("Segregate Waste")

# --- Main Page ---

if uploaded_file is not None:
    # Use uploaded image
    image = Image.open(uploaded_file)
elif selected_example:
    # Use example image
    try:
        response = requests.get(example_images[selected_example], stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading example image: {e}")
        image = None
else:
    image = None
    
# Display the image if available
if image:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption="Image to be processed.", use_column_width=True)

    if run_button:
        if not api_key:
            st.warning("Please enter your Roboflow API key in the sidebar.")
        else:
            with st.spinner('Detecting waste items...'):
                try:
                    # --- Roboflow API Call ---
                    rf = Roboflow(api_key=api_key)
                    project = rf.workspace().project(model_id.split('/')[0])
                    model = project.version(int(model_id.split('/')[1])).model

                    # Save the uploaded image temporarily
                    image.save("temp_image.jpg")
                    
                    # Predict
                    prediction = model.predict("temp_image.jpg", confidence=confidence_threshold * 100, overlap=30).json()

                    # --- Display Results ---
                    if prediction['predictions']:
                        with col2:
                            st.subheader("Detection Results")
                            # Roboflow returns a saved image URL with bounding boxes
                            result_image_url = f"https://inference.roboflow.com/{model.id}/0?api_key={api_key}&image=temp_image.jpg"
                            st.image(f"{prediction['predictions'][0]['image_path']}?api_key={api_key}", caption="Image with Bounding Boxes", use_column_width=True)
                        
                        st.success(f"Detected {len(prediction['predictions'])} items.")
                        
                        # --- Create Graphs ---
                        st.subheader("Analysis of Detected Waste")
                        
                        class_counts = {}
                        for pred in prediction['predictions']:
                            class_name = pred['class']
                            if class_name in class_counts:
                                class_counts[class_name] += 1
                            else:
                                class_counts[class_name] = 1

                        # Bar chart for class counts
                        df = pd.DataFrame(list(class_counts.items()), columns=['Waste Type', 'Count'])
                        
                        fig, ax = plt.subplots()
                        ax.bar(df['Waste Type'], df['Count'], color='skyblue')
                        ax.set_xlabel('Waste Type')
                        ax.set_ylabel('Count')
                        ax.set_title('Waste Segregation Counts')
                        plt.xticks(rotation=45, ha='right')
                        
                        st.pyplot(fig)
                        
                        # Display data as a table
                        st.subheader("Detailed Counts")
                        st.table(df)

                    else:
                        with col2:
                            st.warning("No objects were detected with the given confidence threshold.")
                            
                    # Clean up the temp image
                    if os.path.exists("temp_image.jpg"):
                        os.remove("temp_image.jpg")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
