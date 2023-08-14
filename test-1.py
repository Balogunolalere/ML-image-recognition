import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


model = keras.applications.MobileNetV2(weights='imagenet')


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    return image


def get_image_predictions(image):
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return keras.applications.mobilenet_v2.decode_predictions(predictions)[0]


def extract_tags(predictions):
  results = []
  for pred in predictions:
    label = pred[1]
    prob = round(pred[2] * 100, 2)
    results.append(f"{label} ({prob}%)")
  
  return results


# Provide the path to the image you want to recognize
image_path = '/content/knife.jpeg'

# Load and pre-process the image
preprocessed_image = load_and_preprocess_image(image_path)

# Get predictions from the model
predictions = get_image_predictions(preprocessed_image)

# Extract tags with probabilities 
tags = extract_tags(predictions)

print("Predictions:", tags)