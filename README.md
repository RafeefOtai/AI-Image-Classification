# Image-Classification
# Task 1: AI Track 

## Overview
This project uses Teachable Machine by Google to train an image recognition model with two classes. The trained model was then loaded in a Python script to predict the class of a new image.

## Step 1: Training the Model
I used Teachable Machine to create two classes and uploaded images for each one. After uploading the images, I trained the model and tested it in the preview panel to check the accuracy.

![Training the model](Class1.png)
![Training the model](Class2.png)


## Step 2: Setting up the Environment
The model needed TensorFlow 2.12.1, which is not compatible with the Python version installed on my system. To fix this, I created a new environment using Anaconda with Python 3.10 and installed the required libraries there (opencv-python, tensorflow, pillow, numpy). Then I connected VS Code to this environment so it could run the script correctly.

## Step 3: Python Script
I wrote a Python script that loads the exported model, opens a test image, resizes it to fit the model input size, and predicts its class. The script also prints the predicted class name and the confidence score.

```python
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open(r"C:\Users\rafee\OneDrive\Desktop\converted_keras (1)\Test_image1.png").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
```

## Step 4: Output
I ran the script in VS Code and the model correctly predicted the class of the test image with high confidence.

![Model output](Output.png)
