from PIL import Image
from pathlib import Path
import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

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
    load_model(age_model, model_path="models", model_name="model_age.pt")
    return age_model
    
@st.cache_resource
def load_gen_model():
    gen_model = GENV0(input_shape=1, output_shape=1)
    load_model(gen_model, model_path="models", model_name="model_gen.pth")
    return gen_model
    
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")    

age_model = load_age_model()
gen_model = load_gen_model()
face_detector = load_face_detector()

img = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

button = st.button("Predict")

cols = st.columns(2)
if button and img:

    img = Image.open(img)
    img_arr = np.array(img)
    
    if img_arr.shape[2] == 4:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGRA2BGR)

    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img_gray, 1.3, 5)
    
    with cols[0]:
        if len(faces) == 0:
            st.write("No face detected")
        else:
            for (x, y, w, h) in faces:
                face = img_gray[ y : y + h, x : x + w]

                age = predict_age(face, model=age_model)
                gender = predict_gender(face, model=gen_model)

                st.write(f"Age: {age}\nGender: {gender}")
                st.image(face)
    with cols[1]:
        st.image(img)