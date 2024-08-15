import streamlit as st

st.set_page_config(page_title="Home", layout='wide', page_icon="./images/home.png")
st.title("YOLO V5 Object Detection APP Example")
st.caption('This web app demonstrates Object detection')

# Content
st.markdown("""
### This App detects objects from Images
- Detection from 20 different classes
- [Click here for Image](/YOLO_for_Image)
- [Click here for Video](/YOLO_for_Video)
- [Click here for Real Time Video Detection](/YOLO_for_RealTime_Video)

Classes that this model detects:
1. Person
2. Car
3. Chair
4. Bottle
5. Pottedplant
6. Bird 
7. Dog
8. Sofa
9. Bicycle
10. Horse
11. Boat
12. Motorbike
13. Cat
14. TVmonitor
15. Cow
16. Sheep
17. Aeroplane
18. Train
19. Diningtable
20. Bus

""")