import streamlit as st 
from PIL import Image
import numpy as np
from predict_user import organize_image, predict

st.title("Finding 😺 in pictures")
st.write("### Using a layered neural network to classify images as 'cat' / 'non-cat'.")
st.sidebar.write('# 😺 or 🚫')
uploaded_file = st.sidebar.file_uploader(label = 'upload your image:')

if uploaded_file is not None:

    my_image = organize_image(uploaded_file)
    my_predicted_image, classes = predict(my_image)

    if classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") == 'cat':
        st.write("# The L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" 😺 picture.")
    else:
        st.write("# The L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" 🚫 picture.")
        
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption='Uploaded Image', use_column_width = True)
    st.image('images/model-resume.png', caption = 'L-layer Model', use_column_width = True)