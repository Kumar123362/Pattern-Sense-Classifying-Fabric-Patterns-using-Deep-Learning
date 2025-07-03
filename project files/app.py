import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model and set parameters
model = load_model('fabric_model_resnet50.h5')
img_size = 224
class_names = sorted(os.listdir('dataset'))  # Should match your training folder names

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocess the image
    img = image.load_img(file_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(preds)]
    confidence = round(np.max(preds) * 100, 2)

    return render_template('result.html',
                           filename=filename,
                           prediction=predicted_class,
                           confidence=confidence)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
