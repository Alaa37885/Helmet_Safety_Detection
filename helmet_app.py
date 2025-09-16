import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# Load the YOLO model
model = YOLO("best.pt")  # Update path if needed

st.title("Helmet Detection App 🪖")

# Choose input type
option = st.radio("Select input type:", ("📷 Image", "🎥 Video"))

# ---------------- IMAGE PROCESSING ----------------
if option == "📷 Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Open image and ensure it's RGB
        image = Image.open(uploaded_file).convert("RGB")

        # Run YOLO prediction
        results = model.predict(np.array(image))
        plotted = results[0].plot()

        # Split interface into 2 columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Original Image")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("✅ After YOLO Detection")
            st.image(plotted, use_column_width=True)

# ---------------- VIDEO PROCESSING ----------------
elif option == "🎥 Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        # Split interface into 2 columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎥 Original Video")
            original_placeholder = st.empty()

        with col2:
            st.subheader("✅ After YOLO Detection")
            detected_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Ensure frame is 3 channels (BGR → RGB if needed)
            if frame.shape[2] == 4:  # If frame has alpha channel
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Show original frame
            original_placeholder.image(frame, channels="BGR", use_column_width=True)

            # YOLO detection
            results = model.predict(frame, verbose=False)
            detected_frame = results[0].plot()

            # Show detected frame
            detected_placeholder.image(detected_frame, channels="BGR", use_column_width=True)

        cap.release()
        os.remove(tfile.name)
