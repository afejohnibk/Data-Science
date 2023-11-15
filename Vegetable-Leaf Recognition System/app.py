# Importing libraries
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image


# Load the trained model
model = tf.keras.models.load_model('C:/..../project/trained_model.h5')
class_names = ["Amunututu", "Ariyanfo'gba", "Ebolo", "Efirin", "Elegede", "Ewuro", "Isapa", "Iyanapaja", "Soko", "Ugwu"]  # class names

# Create Flask app
app = Flask(__name__)

# Define a route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']

    # Preprocess the image
    img = preprocess_image(file)

    # Make predictions
    predictions = model.predict(tf.expand_dims(img, axis=0))
    predicted_class = class_names[tf.argmax(predictions[0])]

    # Return the predicted class as JSON
    return jsonify({'class': predicted_class})

def preprocess_image(file):
    # Preprocess the image (e.g., resize, normalize, etc.)
    # Return the preprocessed image as a NumPy array
    # Example:
    img = Image.open(file)
    img = img.resize((64, 64))
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

# Run the Flask app
if __name__ == '__main__':
    app.run()
