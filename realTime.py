from PIL import Image
from pathlib import Path
import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import mediapipe as mp  
import streamlit as st

# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parent))

from ageModular.age_model_builder import model_AGEV0
from genModular.gen_model_builder import GENV0

from ageModular.age_predict import predict_age
from genModular.gen_predict import predict_gender
from genModular.gen_utils import load_model

st.write("Real Time Age and Gender Prediction")

@st.cache_resource
def load_age_model():
    age_model = model_AGEV0(input_shape=1, output_shape=1)
    load_model(age_model, model_path="models", model_name="AGEV0.pth")
    return age_model
    
@st.cache_resource
def load_gen_model():
    gen_model = GENV0(input_shape=1, output_shape=1)
    load_model(gen_model, model_path="models", model_name="GENV0.pt")
    return gen_model
    
@st.cache_resource
def load_face_detector():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    return face_detection

age_model = load_age_model()
gen_model = load_gen_model()
face_detector = load_face_detector()

img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

button = st.button("Predict")

cols = st.columns(2)
if button and img:

    img = Image.open(img)
    img_arr = np.array(img)
    
    ih, iw, ic = img_arr.shape
    
    if img_arr.shape[2] == 4:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2RGB)

    # img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    results = face_detector.process(img_arr)
    # print(results.detections)
    with cols[0]:
        if len(results.detections) == 0:
            st.write("No face detected")
        else:
            for result in results.detections:
                print(result.location_data.relative_bounding_box)
                bbox = result.location_data.relative_bounding_box
                x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                xmin = int(iw * x)
                ymin = int(ih * y)
                width = int(iw * w)
                height = int(ih * h)
                face = img_arr[ ymin : ymin + height, xmin : xmin + width]
                # img = cv2.rectangle(img_arr, (xmin, ymin), (width, height), (255, 0, 0), 2)
                # face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                st.image(face, caption="Detected Face")
                
                age = predict_age(face, model=age_model)
                gender = predict_gender(face, model=gen_model)

                st.write(f"Age: {age}\nGender: {gender}")
    with cols[1]:
        st.image(img)
        
        
