import streamlit as st
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import cv2

st.set_page_config(
    page_title="Neural Style Transfer", layout="wide"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

tf.executing_eagerly()


def load_image(image_buffer, image_size=(512, 256)): 
    img = image_buffer.astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def export_image(tf_img):
    pil_image = Image.fromarray(np.squeeze(tf_img*255).astype(np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    byte_image = buffer.getvalue()
    return byte_image

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return image
    else:
        return None

def st_ui():
    col1, col2, col3 = st.columns(3)
    
    st.sidebar.title(" Artistic Style Transfer")
    st.sidebar.markdown("Your personal neural style transfer")
    
    # Option to select camera or file upload for content image
    col1.header("Content Image")
    content_option = col1.radio("Select Image Source", ("Camera", "File Upload"), key="content_option")

    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None

    content_image = None  # Initialize content_image variable

    if content_option == "Camera":
        if col1.button("Capture Image"):
            st.session_state.captured_image = capture_image()
            if st.session_state.captured_image is not None:
                content_image = load_image(st.session_state.captured_image)
        if st.session_state.captured_image is not None:
            if content_image is None:
                content_image = load_image(st.session_state.captured_image)
            col1.image(st.session_state.captured_image, use_column_width=True)
    else:
        image_upload1 = st.file_uploader("Upload your content image here", type=["jpeg", "png", "jpg"], accept_multiple_files=False, key=None, help="Upload the image you want to style")

        if image_upload1 is not None:
            content_image = load_image(np.array(Image.open(image_upload1)))
            col1.image(image_upload1, use_column_width=True)
    
    # File upload for style image
    image_upload2 = st.sidebar.file_uploader("Upload your style image here", type=["jpeg", "png", "jpg"], accept_multiple_files=False, key=None, help="Upload the image whose style you want")

    if image_upload2 is not None:
        style_image = load_image(np.array(Image.open(image_upload2)))
        style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
        col2.header("Style Image")
        col2.image(image_upload2, use_column_width=True)
    else:
        default_style_image_path = "vangogh.jpg"
        default_style_image = Image.open(default_style_image_path)
        default_style_image_np = np.array(default_style_image)
        style_image = load_image(default_style_image_np)
        style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
        col2.header("Default Style Image")
        col2.image(default_style_image, use_column_width=True)

    if st.sidebar.button(label="Start Styling"):
        with st.spinner('Generating Stylized image ...'):
            # Load image stylization module.
            stylize_model = tf.saved_model.load("C:/Users/INSHA/Downloads/arbitrary-image-stylization-v1-tensorflow1-256-v2")

            if content_image is not None:  # Check if content_image is not None
                results = stylize_model(tf.constant(content_image), tf.constant(style_image))
                stylized_photo = results[0]
                col3.header("Final Image")
                col3.image(np.array(stylized_photo))
                st.download_button(label="Download Final Image", data=export_image(stylized_photo), file_name="stylized_image.png", mime="image/png")
            else:
                st.sidebar.error("Please upload a content image or capture one from the camera.")

if __name__ == "__main__":
    st_ui()
