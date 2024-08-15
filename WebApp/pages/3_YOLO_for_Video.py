import streamlit as st
from yolo_predictions_Yeison import YOLO_Pred
from PIL import Image
import numpy as np
import cv2
import tempfile


st.set_page_config(page_title="YOLO Object Detection for Video",
                   layout='wide',
                   page_icon='./images/object.png')

st.title('YOLO Object Detection for Videos')

st.write('Please upload a Video to get detections')

# Import the model
with st.spinner('Please wait while the model is loading...'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx',
                    data_yaml='./models/data.yaml')
    
    # st.balloons()
def upload_video():
    # Upload Image
    video_file = st.file_uploader(label='Upload Video')
    if video_file is not None:
        size_mb = video_file.size/(1024*1024)
        file_details = {"filename": video_file.name,
                        "filetype": video_file.type,
                        "filesize": "{:,.2f} MB".format(size_mb)}
        
        #Validate that we got a video file
        if file_details["filetype"] in ('video/mp4'):
            st.success('Valid Video file type')
            return {"file": video_file, "details": file_details}
        else:
            st.error('Invalid file format')
            st.error('Upload only .mp4')
            return None
        
def main():
    object = upload_video()
    
    if object:
        
        # prediction = False
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(object['file'].read())
        cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                print('unable to read video')
                break
            
            try:
                pred_image = yolo.predictions(frame)
            except:
                pass
            color = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
            stframe.image(color)

if __name__ == "__main__":
    main()