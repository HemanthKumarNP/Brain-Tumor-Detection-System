from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras import backend as K
import numpy as np
from flask_cors import CORS
from keras.utils import load_img
from keras.utils import img_to_array
from langchain_google_genai import ChatGoogleGenerativeAI
from collections import defaultdict
from langchain_core.messages import HumanMessage
import os
from PIL import Image
# from IPython.display import Image, display
import json

import pathlib
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from pathlib import Path
from langchain_core.messages import HumanMessage
os.environ["GOOGLE_API_KEY"] = "Your Gemini Api Key"

llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = 'models/model2.h5'
model = load_model(model_path)



# Initialize annotated_dict
annotated_dict = defaultdict(str)

# Sample image paths
image_ls = ["temp.jpg"]

# Grad-CAM function
def grad_cam(model, img_array, layer_name):
    cls = np.argmax(model.predict(img_array))
    class_output = model.output[:, cls]
    last_conv_layer = model.get_layer(layer_name)
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img_array])
    for i in range(pooled_grads.shape[0]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Define a function to preprocess the uploaded image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(240, 240))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = imagenet_utils.preprocess_input(img_array)
    return img_array

# Define a route to handle Gemini data
@app.route('/gemapi', methods=['GET'])
def getGeminiData():
    try:
        for i in image_ls:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Provide findings and recommended treatment for the following MRI image: {image_url}."},
                    {"type": "image_url", "image_url": i},
                ]
            )
            response = llm.invoke([message])
            annotated_dict[i] = response

        # Retrieve data from annotated_dict
        results = []
        for k, v in annotated_dict.items():
            # Check if v has a 'content' attribute
            if hasattr(v, 'content'):
                # Access the 'content' attribute
                content = v.content
                # Append data to results
                results.append({'File name': k, 'Description': content})
            else:
                results.append({'File name': k, 'Description': 'No content available'})

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})


# Define a route to handle image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if a file is included in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No file included'})

        # Get the uploaded file
        file = request.files['image']

        # Save the uploaded file to a temporary location
        file_path = 'temp.jpg'
        file.save(file_path)

        # Preprocess the uploaded image
        processed_img = preprocess_image(file_path)

        # Make predictions
        prediction = model.predict(processed_img)

        # Convert prediction to human-readable format
        class_dict = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}
        tumor_type = class_dict[np.argmax(prediction)]
        print(tumor_type)

        return jsonify({'status': True, 'prediction': tumor_type})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

