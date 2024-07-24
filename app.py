import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

def main():
    st.title('Image Prediction App')
    st.write('Upload an image for prediction')

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Display the image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        resized_image = cv2.resize(img, (256, 256))
        expanded_image = np.expand_dims(resized_image, axis=0)
        model_path = '/content/models/imageclassifier.h5'
        model = load_model(model_path)
        # Make predictions
        if st.button('Predict'):
            predictions = model.predict(expanded_image)
            st.write(f'Prediction: {predictions}')
if __name__ == '__main__':
    main()