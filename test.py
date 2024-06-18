from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.utils import load_img
from keras.utils import img_to_array

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(240, 240))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load the trained model
model_path = 'models/Brain.h5'
model = load_model(model_path)

# Path to the sample image
sample_image_path = 'temp.jpg'  # Adjust the path as per your file structure

# Preprocess the sample image
sample_img = preprocess_image(sample_image_path)

# Perform prediction
predictions = model.predict(sample_img)

# Convert prediction to human-readable format
class_dict = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}
predicted_class = class_dict[np.argmax(predictions)]

print("Predicted class:", predicted_class)
