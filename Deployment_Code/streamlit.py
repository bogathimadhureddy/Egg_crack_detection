# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🥚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation

model_path = r"C:\Users\madhu\OneDrive\Desktop\Egg_crack_project\deploy\weights\best.pt"

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]
source_radio = st.sidebar.radio(
    "Select Source", SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == 'Image':
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                st.markdown(
                    """
        <div style="background-color: #FFDDC1; padding: 10px; border-radius: 5px; text-align: center;">
            <h3 style="color: #D9534F;">Please Upload the Image</h3>
        </div>
        """,
                    unsafe_allow_html=True
                )
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if st.sidebar.button('Detect Objects'):
            res = model.predict(uploaded_image,
                                conf=confidence
                                )
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                     use_column_width=True)
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                # st.write(ex)
                st.write("No image is uploaded yet!")

elif source_radio == 'Video':
    helper.video(confidence, model)

elif source_radio == 'Webcam':
    helper.play_webcam(confidence, model)

else:
    st.error("Please select a valid source type!")
