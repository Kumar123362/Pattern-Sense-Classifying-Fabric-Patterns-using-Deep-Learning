import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Parameters
img_size = 224
model_path = 'fabric_model_resnet50.h5'
class_names = sorted(os.listdir('dataset'))  # Assuming 1 folder per class

# Load the trained model
model = load_model(model_path)

# Function to predict single image
def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.show()

# Example usage
img_to_test = 'download (4).jpeg'  # Replace with your image path
predict_image(img_to_test)