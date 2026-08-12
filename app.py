"""
Streamlit app for YOLOv8 Vehicle Detection
-------------------------------------------
Deploys the model trained in `Yolo_v8_segmentation_on_traffic_vehicles.ipynb`
(single class: "Vehicle") for inference on images and video via a web UI.

Run locally:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLOv8 Vehicle Detection",
    page_icon="🚗",
    layout="wide",
)

MODEL_PATH = Path("best.pt")


# --------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def get_model():
    if not MODEL_PATH.exists():
        st.error(
            f"Model weights not found at `{MODEL_PATH}`.\n\n"
            "Export `best.pt` from your training run "
            "(`runs/detect/train/weights/best.pt`) and place it in the "
            "`weights/` folder before deploying, or update `MODEL_PATH` in "
            "`app.py`."
        )
        st.stop()
    return load_model(str(MODEL_PATH))


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU threshold (NMS)", 0.0, 1.0, 0.45, 0.05)
source_type = st.sidebar.radio("Input type", ["Image", "Video"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: YOLOv8n fine-tuned for vehicle detection (1 class: `Vehicle`)."
)

# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------
st.title("🚗 YOLOv8 Traffic Vehicle Detection")
st.write(
    "Upload an image or video and the model will draw bounding boxes "
    "around detected vehicles."
)

model = get_model()

if source_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        with st.spinner("Running inference..."):
            results = model.predict(
                source=np.array(image),
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
            annotated = results[0].plot()  # BGR numpy array
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("Detections")
            st.image(annotated_rgb, use_container_width=True)

        n_detections = len(results[0].boxes)
        st.success(f"Detected {n_detections} vehicle(s).")

        if n_detections > 0:
            st.subheader("Detection details")
            rows = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                rows.append(
                    {
                        "class": model.names[cls_id],
                        "confidence": round(float(box.conf[0]), 3),
                        "bbox_xyxy": [round(v, 1) for v in box.xyxy[0].tolist()],
                    }
                )
            st.dataframe(rows, use_container_width=True)

else:  # Video
    uploaded_file = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:
        # Persist upload to a temp file so OpenCV/YOLO can read it
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
        tfile.write(uploaded_file.read())
        input_path = tfile.name

        st.video(input_path)

        if st.button("Run detection on video"):
            output_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            progress = st.progress(0, text="Processing video...")
            frame_idx = 0

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )
                annotated = results[0].plot()
                writer.write(annotated)

                frame_idx += 1
                if total_frames > 0:
                    progress.progress(
                        min(frame_idx / total_frames, 1.0),
                        text=f"Processing frame {frame_idx}/{total_frames}",
                    )

            cap.release()
            writer.release()
            progress.empty()

            st.success("Done! Annotated video below.")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    "Download annotated video",
                    data=f,
                    file_name="annotated_output.mp4",
                    mime="video/mp4",
                )

st.markdown("---")
st.caption("Built with Streamlit + Ultralytics YOLOv8.")
