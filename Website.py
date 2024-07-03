import streamlit as st
import cv2
import numpy as np
import time
import tensorflow as tf
from keras.models import load_model
from PIL import Image

d = {26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8',
     35: '9', 0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
     8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
     17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

try:
  model = load_model('model.h5')
except OSError as e:
  st.error(f"Error loading model: {e}")
  exit(1)  # Exit the app if model loading fails

st.header("Handwritten Digit and Charecter Recognization using CNN")
uploaded_file = st.file_uploader("Upload any Digit or Charecter Image of 28*28 pixel resolution")
if uploaded_file:
  image_file = Image.open(uploaded_file)
  image = image_file.resize((28, 28))

  def read_image(image):
    pic = np.array(image)
    if len(pic.shape) > 2:
      pic = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
    pic = np.invert(pic)
    pic = np.expand_dims(pic, axis=0)  # Add batch dimension
    pic = pic.astype('float32') / 255.0  # Normalize pixel values (optional, adjust based on your model)
    # No need to compile the model here (remove the compile line)
    try:
      prediction = model.predict(pic)
      return np.argmax(prediction)
    except Exception as e:  # Catch any potential errors during prediction
      st.error(f"Error during prediction: {e}")
      return None  # Indicate failure

  if st.button("Show Predicted value"):
    with st.spinner("LOADING....."):
      time.sleep(1)  # Reduce or remove the delay if not needed
      c3, c4 = st.columns(2)
      with c3:
        st.write("Uploaded Image:")
        resized_image = image.resize((128, 128))
        st.image(resized_image)
      with c4:
        predicted_char = read_image(image)
        if predicted_char is not None:  # Check if prediction was successful
          k = d[predicted_char]
          st.markdown(f'The Predicted value is :')
          st.subheader(k)
