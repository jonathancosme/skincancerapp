#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 03:01:24 2021

@author: jcosme
"""

import numpy as np
import streamlit as st
#import tensorflow as tf
from funcs import *
from PIL import Image


st.markdown("""
            # Skin Growth Classifier*  
            """
            )

st.markdown("""
            This app will classify a user-uploaded photo (presumably of a skin growth) as either 'Benign' or 'Malignant.'  
              ## Steps:  
            1. Take a photo of a skin growth (make sure it's VERY close-up; see "Details" section below).  
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
    img = np.array([img.astype(np.float32)])
    prob = get_prob(img)
    if prob <= 0.5:
        prediction = 'Benign'
    else:
        prediction = 'Malignant'
    st.markdown('## ' + prediction + '!')

st.markdown("""
            ### More Info  
            This app uses a convolutional neural network built with Tensorflow, using this [dataset](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign).  
            It achieved around 82% accuracy on both the training, and test, datasets.
            """
            )

st.markdown("""
            # *This app should NOT be relied upon for any medical diagnosis.   
            ### Please consult a physician for any concerns you may have.
            """
            )

st.markdown("""
            ### Details 
            If you take a photo that is too far away, the algo will not classify it correctly. 
            """ )
st.markdown("""for example, this one is too far away, and get incorrectly classified as "malignant"  
            """ )
st.image('./images/wrong.png', width=200)
st.markdown("""when the photo is taken closer (or copped) it is correctly classified as "benign"  
            """ )
st.image('./images/correct.png', width=200)
st.markdown("""Apparently cats are classified as "malignant"  
            """ )
st.image('./images/malignantCat.png', width=200)
