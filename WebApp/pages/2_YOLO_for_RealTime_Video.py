import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from yolo_predictions_Yeison import YOLO_Pred

# Load YOLO Model
yolo = YOLO_Pred(onnx_model='./models/best.onnx',
                 data_yaml='./models/data.yaml')

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    pred_img = yolo.predictions(img)

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="example",
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video":True, "audio": False})