import cv2
import streamlit as st
from detect import detect, arm_angle, arm_lift_angle
import random
import numpy as np
from PIL import Image, ImageFont
from PIL.ImageDraw import Draw

cap = cv2.VideoCapture(0)

st.title("FIT.LY")
stframe = st.empty()

class Resetter():
    def __init__(self):
        self.button_clicked = False

    def on_button_click(self):
        self.button_clicked = True

resetter = Resetter()

if st.button('Reset'):
    resetter.on_button_click()

values = ['Curls', 'Shoulder Press', 'Lateral Raises']

# Display the dropdown menu and get the user's selection
selection = st.selectbox('Select a value', values)


class curls:
    def  __init__(self):
        self.state = -1
        self.reps = 0
    
    def reset(self):
        self.reps = 0 
    # !state ensures doesn't change from both to one if moving arms slightly out of sync

    def curls(self, L, R):
        '''
        CURL STATE

        0 - both extended
        1 - R extended
        2 - L extended
        3 - R bent
        4  - L bent
        5 - both bent
    '''
        if L < 60 and R < 60: # Both Bent
            return 5
        if L > 90 and R > 90: # Both Extend
            return 0
        if L > 90 and self.state != 0: # Left Extend
            return 2
        if R > 90 and self.state != 0: # Right Extend
            return 1
        if L < 60 and self.state != 5: # Left Bent
            return 4
        if R <60 and self.state != 5: # Right Bent
            return 3
        return self.state

    def countCurl(self, prev, curr):
        '''
        CURL COUNTER

        0 - none
        1 - rep

        '''

        if curr == 4: # Left Bent
            if prev == 0 or prev == 2: # If previously extended
                return 1
        if curr == 3: # Right Bent
            if prev == 0 or prev == 1: # If previously extended
                return 1
        if curr == 5 and prev == 0: # Both Bent; Previously both extended
                return 1
        
        return 0
        # if curr == 0 and prev == 5: # Both Bent; Previously both extended
        #         return 0
        
        # if curr == 0 or curr == 2: # Left Extended
        #     if prev == 4: # If previously bent
        #         return 0
        
        # if curr == 0 or curr == 1: # Right Extended
        #     if prev == 3: # If previously bent
        #         return 0
        
        # return 0
    
    def run(self, LAngle, RAngle, LArm, RArm):
        if resetter.button_clicked:
            self.reset()
            resetter.button_clicked = False
        new = self.curls(LAngle, RAngle)
        output = self.countCurl(self.state, new)
        if output:
            self.reps += 1
        self.state = new

class raises:
    def  __init__(self):
        self.state = -1
        self.reps = 0
    
    # !state ensures doesn't change from both to one if moving arms slightly out of sync

    def raises(self, L, R):
        '''
        LAT RAISE STATE

        0 - both unextended
        1 - R unextended
        2 - L unextended
        3 - R extended
        4  - L extended
        5 - both extended
    '''
        if L < 30 and R < 30: # Both Bent
            return 5
        if L > 80 and R > 80: # Both Extend
            return 0
        if L > 80 and self.state != 0: # Left Extend
            return 2
        if R > 80 and self.state != 0: # Right Extend
            return 1
        if L < 30 and self.state != 5: # Left Bent
            return 4
        if R <30 and self.state != 5: # Right Bent
            return 3
        return self.state

    def bentArm(self, RArm, LArm):
        if RArm <165 or LArm <165:
            self.state = 0

    def countLatRaise(self, prev, curr):
        '''
        LAT RAISE COUNTER

        0 - none
        1 - rep

        '''

        if curr == 4: # Left Bent
            if prev == 0 or prev == 2: # If previously extended
                return 1
        if curr == 3: # Right Bent
            if prev == 0 or prev == 1: # If previously extended
                return 1
        if curr == 5 and prev == 0: # Both Bent; Previously both extended
                return 1
        
        return 0
        # if curr == 0 and prev == 5: # Both Bent; Previously both extended
        #         return 0
        
        # if curr == 0 or curr == 2: # Left Extended
        #     if prev == 4: # If previously bent
        #         return 0
        
        # if curr == 0 or curr == 1: # Right Extended
        #     if prev == 3: # If previously bent
        #         return 0
        
        # return 0
    
    def run(self, LAngle, RAngle, LArm, RArm):
        self.bentArm(RArm, LArm)

        new = self.raises(LAngle, RAngle)
        output = self.countLatRaise(self.state, new)
        if output:
            self.reps += 1
        self.state = new


def app(count = 0):
    if selection == "Curls":
        evaluator = curls()
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
            LArm, RArm = arm_lift_angle(kp, frame.shape[0], frame.shape[1])
            draw.text((10, 10), f"Left Arm: {round(LAngle)}", font=font, fill =(0, 0, 255))
            draw.text((10, 50), f"Right Arm: {round(RAngle)}", font=font, fill =(0, 0, 255))
            
        evaluator.run(LAngle, RAngle, LArm, RArm)

        draw.text((10, 90), f"Reps: {evaluator.reps}", font=font, fill =(0, 0, 255))

        cv2.imshow('frame', np.array(out)) 
        
        stframe.image(out)



    cap.release()
    cv2.destroyAllWindows()

# Run the Streamlit app
if __name__ == '__main__':
    app()