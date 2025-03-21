from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import torch
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load the trained model
model = load_model('model/efficientnet_model.h5')

# Load the segmentation model
segmentation_model = torch.load('trained_model.pth', map_location=torch.device('cpu'))

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # adjust this to match your model's expected input size
    image = image.convert('RGB')
    image = torch.tensor(np.array(image)) / 255.0  # normalize to [0, 1]
    image = image.unsqueeze(0)  # add batch dimension
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Predict
        pred = model.predict(img_array)
        classes = ['Normal', 'Benign', 'Malignant']
        result = classes[np.argmax(pred)]
        
        return render_template('index.html', prediction=result, image_path=filepath)

@app.route('/segment', methods=['POST'])
def segment_image():
    image_file = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    # Preprocess the input image
    image = preprocess_image(image_path)

    # Run the input image through the model
    output = segmentation_model(image)

    # Process the output (e.g., convert to a mask, threshold, etc.)
    # ...

    # Return the output as a response
    return jsonify({'output': output.tolist()})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
    