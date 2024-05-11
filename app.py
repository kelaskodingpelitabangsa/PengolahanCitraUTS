#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt

def rgb_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv_image

def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

def adjust_brightness_contrast(image, brightness, contrast):
    brightness = int((brightness - 0.5) * 255)
    contrast = int(contrast * 255)
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted

def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Handling different OpenCV versions
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return contours

# Main function
def main():
    st.title("Image Manipulation App")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        st.subheader("Original Image")
        st.image(image, channels="RGB", use_column_width=True)
        
        # Convert RGB to HSV
        hsv_image = rgb_to_hsv(image)
        st.subheader("HSV Image")
        st.image(hsv_image, channels="HSV", use_column_width=True)
        
        # Calculate Histogram
        hist = calculate_histogram(image)
        plt.plot(hist)
        st.subheader("Histogram")
        st.pyplot(plt)
        
        # Adjust Brightness and Contrast
        brightness = st.slider("Brightness", 0.0, 1.0, 0.5)
        contrast = st.slider("Contrast", 0.0, 2.0, 1.0)
        adjusted_image = adjust_brightness_contrast(image, brightness, contrast)
        st.subheader("Adjusted Image")
        st.image(adjusted_image, channels="RGB", use_column_width=True)
        
        # Find Contours
        contours = find_contours(image)
        contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
        st.subheader("Contour Image")
        st.image(contour_image, channels="RGB", use_column_width=True)

if __name__ == "__main__":
    main()

