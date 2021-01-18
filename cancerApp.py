#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 03:01:24 2021

@author: jcosme
"""

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os
MODEL_PATH = '/app/cancer.keras'


@st.cache(allow_output_mutation=True)
def load_kr_model():
    model = load_model(MODEL_PATH)
    return model

st.markdown("""
            # Skin Growth Classifier*  
            """
            )

st.markdown("""
            This app will classify a user-uploaded photo (presumably of a skin growth) as either 'Benign' or 'Malignant.'  
              ### Steps:  
            1. Take a photo of a skin growth (make sure it's VERY close-up).  
            2. Upload the photo  
            3. Select the 'Malignant or Benign?' button  
            """
            )

uploaded_file = st.file_uploader("Step 1: Select a photo to upload.")

if st.button('Malignant or Benign?'):
    img = Image.open(uploaded_file)
    img = img.resize((224,224))
    img = np.array(img)
    st.image(img, channels="RGB")
    model= load_kr_model()
    prob = model(np.array([img]), training=False)[0]
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
