import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2

# Load and preprocess the image
def model_predict(image_path):
    model = tf.keras.models.load_model('plant_disease_cnn_model.keras')
    img = cv2.imread(image_path)
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img.astype('float32')
    img = img / 255.0
    img = img.reshape(1, H, W, C)

    prediction = np.argmax(model.predict(img), axis=-1)[0]

    return prediction

st.sidebar.title('Plant Disease Prediction System for Sustainable Agriculture')
app_mode = st.sidebar.selectbox('Select page',['Home', 'Disease Recognition'])

from PIL import Image
img = Image.open('Disease.png')
st.image(img)

if(app_mode == 'Home'):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Prediction System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif(app_mode == 'Disease Recognition'):
    st.header("Plant Disease Prediction System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        save_path = os.path.join(os.getcwd(), 'test_image.name')
        print(save_path)
        with open(save_path, 'wb') as f:
            f.write(test_image.getbuffer())

    if(st.button("Show Image")):
        st.image(test_image, width=4, use_container_width=True)

    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_predict(save_path)
        print(result_index)

        class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                    'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew', 
                    'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy', 
                    'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 
                    'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy', 
                    'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy', 
                    'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew', 
                    'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot', 
                    'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold', 
                    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', 
                    'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                      'Tomato___healthy']
    
        st.success("Model is predicting that it is a {}".format(class_name[result_index]))