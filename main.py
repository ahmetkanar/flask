import datetime

from flask import Flask, request, jsonify
import cv2
import numpy as np
import keras
from flask_cors import CORS
from datetime import date
import os

app = Flask(__name__)
CORS(app)


def improved_canny_algorithm(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Resize the image
    img = cv2.resize(img, (500, 500))

    # Filter image
    filtered_img = cv2.medianBlur(img, 15)

    # Adjust image intensity
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    # which enhances the contrast of images by redistributing the
    # intensity values over a specified range.
    clahe = cv2.createCLAHE(clipLimit=1.85, tileGridSize=(7, 7))
    equalized = clahe.apply(filtered_img)

    # Black and white filter
    bw = cv2.threshold(equalized, 100, 255, cv2.THRESH_BINARY)[1]

    # XOR on image
    xor = cv2.bitwise_xor(bw, 255)

    # define canny parameters
    # image
    # Lower Threshold
    # Upper threshold
    # Aperture size
    # Boolean
    edges = cv2.Canny(xor, 80, 130, apertureSize=5, L2gradient=True)

    # Create structuring element for dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Apply dilation and erosion to enhance edges
    dilated = cv2.dilate(edges, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=3)

    # return final image(eroded)
    return cv2.resize(eroded, (224, 224))


def load_image(path):
    edges = improved_canny_algorithm(path)
    img_array = keras.preprocessing.image.img_to_array(edges)
    return img_array


@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['image']
    image_id = request.form['image_id']
    # Save the uploaded image temporarily
    image_path = '/tmp/temporarily.jpg'
    file.save(image_path)

    # Process the image using the provided code
    img = np.array([load_image(image_path)])
    model = keras.models.load_model('model_with_10000_4.h5')
    predictions = model.predict(img)

    # Get the highest label and corresponding values
    labels = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    full_names = ['Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis',
                  'Dermatofibroma', 'Vascular lesion', 'Squamous cell carcinoma', 'Unknown']
    print(predictions)
    highest_index = np.argmax(predictions)
    highest_value = predictions.flatten()[highest_index]
    highest_label = labels[highest_index]
    highest_full_name = full_names[highest_index]

    # Delete the temporary image file
    os.remove(image_path)

    result = {
        'highest_value': float(highest_value),
        'corresponding_full_name': highest_full_name,
        'image_id': image_id,
        'created_date': date.today()
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run()
