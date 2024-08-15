import streamlit as st
from yolo_predictions_Yeison import YOLO_Pred
from PIL import Image
import numpy as np


st.set_page_config(page_title="YOLO Object Detection for Image",
                   layout='wide',
                   page_icon='./images/object.png')

st.title('YOLO Object Detection for Images')

st.write('Please upload an Image to get detections')

# Import the model
with st.spinner('Please wait while the model is loading...'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx',
                    data_yaml='./models/data.yaml')
    
    # st.balloons()
def upload_image():
    # Upload Image
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        size_mb = image_file.size/(1024*1024)
        file_details = {"filename": image_file.name,
                        "filetype": image_file.type,
                        "filesize": "{:,.2f} MB".format(size_mb)}
        # st.json(file_details)
        #Validate that we got an image file
        if file_details["filetype"] in ('image/png', 'image/jpeg'):
            st.success('Valid Image file type')
            return {"file": image_file, "details": file_details}
        else:
            st.error('Invalid file format')
            st.error('Upload only .png or .jpeg')
            return None
        
def main():
    object = upload_image()
    
    if object:
        prediction = False
        image_obj = Image.open(object['file'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info('Image Preview')
            st.image(image_obj)
            
        with col2:
            st.subheader('Image details')
            st.json(object['details'])
            button = st.button('Get Detections from YOLO')
            if button:
                with st.spinner("""
                                Getting Objects from image. Please wait
                                """):
                    # convert the image into numpy array
                    image_array = np.array(image_obj)
                    pred_img = yolo.predictions(image_array)
                    # convert back to image
                    pred_img_obj = Image.fromarray(pred_img)
                    # Display the image
                    prediction = True
        if prediction:
            st.subheader("Predicted Image")
            st.caption("Object detection from YOLO V5")
            st.image(pred_img_obj)

if __name__ == "__main__":
    main()