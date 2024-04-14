import cv2
import streamlit as st
from detect import detect, arm_angle
import random
import numpy as np
from PIL import Image, ImageFont
from PIL.ImageDraw import Draw

cap = cv2.VideoCapture(0)

st.title("FIT.LY")
stframe = st.empty()

values = ['Curls', 'Shoulder Press', 'Lateral Raises']

# Display the dropdown menu and get the user's selection
selection = st.selectbox('Select a value', values)

state = 0
count = 0

def curls(L, R):
    pass

def app():
    while True:
        ret, frame = cap.read()
        out, kp = detect(frame)
        out = np.ascontiguousarray(out, dtype=np.uint8)
        # out = out[:,:,::-1]
        out = Image.fromarray(out)
        draw = Draw(out)
        draw.rectangle((0,0,150,130), fill=(255,255,255))
        font = ImageFont.truetype("arial.ttf", 20)
        if kp is not None:
            LAngle, RAngle = arm_angle(kp, frame.shape[0], frame.shape[1])
            draw.text((10, 10), f"Left Arm: {round(LAngle)}", font=font, fill =(0, 0, 255))
            draw.text((10, 50), f"Right Arm: {round(RAngle)}", font=font, fill =(0, 0, 255))
            
        draw.text((10, 90), f"Reps: {count}", font=font, fill =(0, 0, 255))

        cv2.imshow('frame', np.array(out)) 
        
        stframe.image(out)



    cap.release()
    cv2.destroyAllWindows()

# Run the Streamlit app
if __name__ == '__main__':
    app()