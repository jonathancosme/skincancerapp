#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 03:01:24 2021

@author: jcosme
"""

import streamlit as st
import cv2
import numpy as np
import scipy as sp
from tensorflow.keras.models import load_model
import os

st.markdown("""
            # Skin Growth Classifier*  
            """
            )

st.markdown("""
            This app will classify a user-uploaded photo (presumably of a skin growth) as either 'Benign' or 'Malignant.'  
              ### Steps:  
            1. Upload a photo  
            2. Select the 'Malignant or Benign?' button  
            """
            )

uploaded_file = st.file_uploader("Step 1: Select a photo to upload.")

if st.button('Malignant or Benign?'):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB) 
    opencv_image = cv2.resize(opencv_image, (224, 224))
    st.image(opencv_image, channels="RGB")
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    model = load_model('cancer.keras')
    prob = model(opencv_image.reshape(1, 224, 224, 3), training=False)[0]
    if prob <= 0.5:
        prediction = 'Benign'
    else:
        prediction = 'Malignant'
    st.markdown('## ' + prediction + '!')

st.markdown("""
            #### More Info  
            This app uses a convolutional neural network built with Tensorflow, using this [dataset](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign).  
            It achieved around 82% accuracy on both the training, and test, datasets.
            """
            )

st.markdown("""
            *This app should NOT be relied upon for any medical diagnosis.   
            Please consult a physician for any concerns you may have.
            """
            )
