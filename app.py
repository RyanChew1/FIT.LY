import cv2
import streamlit as st
from detect import detect
import random

# Initialize the webcam
cap = cv2.VideoCapture(0)
stframe = st.empty()# Define the Streamlit app
def app():
    # Run the webcam stream
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Display the frame in Streamlit
        stframe.image(detect(frame))


    # Release the webcam and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the Streamlit app
if __name__ == '__main__':
    app()