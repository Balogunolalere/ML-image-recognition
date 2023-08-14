from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load model
model = ResNet50(weights='imagenet')

# Image loading and preprocessing
def load_and_prep_image(img_path):

  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  
  return x

# Prediction function  
def predict(img):
  preds = model.predict(img)
  return decode_predictions(preds, top=5)[0]

# Extract tags 
def get_tags(predictions):
  results = []
  for pred in predictions: 
     results.append(pred[1])
  return results

# Test on image
img_path = 'sheep.jpg'
x = load_and_prep_image(img_path)
preds = predict(x) 

tags = get_tags(preds)
print("Predicted tags:", tags)