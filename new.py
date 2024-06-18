from keras.models import load_model
from keras.preprocessing import image
from keras.applications import imagenet_utils
import numpy as np
import gradio as gr

# Load the trained model
model_path = 'models/Brain.h5'
model = load_model(model_path)

# Define a function to preprocess the uploaded image
def preprocess_image(img):
    try:
        img = img.resize((240, 240))  # Resize the image to match the input size of the model
        img_array = np.array(img)  # Convert the image to a numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
        img_array = imagenet_utils.preprocess_input(img_array)  # Preprocess the image
        return img_array
    except Exception as e:
        print("Error preprocessing image:", e)
        return None


def predict(img):
    try:
        # Preprocess the uploaded image
        processed_img = preprocess_image(img)
        if processed_img is None:
            return "Error: Image preprocessing failed"

        # Make predictions
        prediction = model.predict(processed_img)
    
        # Convert prediction to human-readable format
        class_dict = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}
        tumor_type = class_dict[np.argmax(prediction)]
        
        # Return the prediction
        return tumor_type
        
    except Exception as e:
        print("Prediction error:", e)
        return "Error making prediction"

# Gradio Interface
inputs = gr.Image(type='pil', label="Upload Brain Image")
outputs = gr.Textbox(label="Tumor Prediction")
title = "Brain Tumor Classifier"
description = "Upload an image of a brain scan and the model will predict the type of tumor present."
examples = [["temp.jpg"]]
iface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title=title, description=description, examples=examples)
iface.launch()
