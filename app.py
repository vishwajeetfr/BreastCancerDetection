import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Multiply, Add, Activation
import tensorflow.keras.backend as K
from werkzeug.utils import secure_filename

# Define Attention Gate Layer
class AttentionGate(Layer):
    def __init__(self, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_g = Conv2D(input_shape[-1], kernel_size=1, strides=1, padding='same', use_bias=False)
        self.W_x = Conv2D(input_shape[-1], kernel_size=1, strides=1, padding='same', use_bias=False)
        self.psi = Conv2D(1, kernel_size=1, strides=1, padding='same', activation='sigmoid', use_bias=True)

    def call(self, inputs):
        g, x = inputs
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = Activation('relu')(Add()([g1, x1]))
        psi = self.psi(psi)
        return Multiply()([x, psi])

    def get_config(self):
        config = super(AttentionGate, self).get_config()
        return config

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Path to the model
MODEL_PATH = "model/Attention_UNet_BreastCancer.h5"

# Load the model with custom layers
custom_objects = {"AttentionGate": AttentionGate}
segmentation_model = load_model(MODEL_PATH, custom_objects=custom_objects)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to preprocess images
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def analyze_prediction(prediction):
    # Get the segmentation mask
    pred_mask = prediction[..., 0]
    
    # Calculate metrics
    total_pixels = pred_mask.size
    threshold = 0.5  # Adjusted threshold for better discrimination
    highlighted_pixels = np.sum(pred_mask > threshold)
    highlighted_percentage = (highlighted_pixels / total_pixels) * 100
    
    # Calculate intensity-based metrics
    max_intensity = np.max(pred_mask)
    mean_intensity = np.mean(pred_mask)
    
    # Calculate connected components for blob analysis
    binary_mask = (pred_mask > threshold).astype(np.uint8)
    num_features = np.sum(binary_mask > 0)
    
    # Normal tissue characteristics
    if (highlighted_percentage < 5.0 and 
        max_intensity < 0.6 and 
        mean_intensity < 0.25 and 
        num_features < 100):
        return "Normal", "No significant abnormalities detected. Continue routine screening as recommended."
    
    # Benign characteristics - improved thresholds for benign detection
    # Relaxed benign thresholds: allow higher highlighted percentage, max and mean intensity.
    elif (highlighted_percentage < 20.0 and 
          max_intensity < 0.85 and 
          mean_intensity < 0.5):
        return "Benign", "Minor abnormalities detected. Follow-up with healthcare provider recommended."
    
    # Malignant characteristics
    else:
        severity = "high" if (max_intensity > 0.9 or highlighted_percentage > 25) else "moderate"
        message = (
            "Significant abnormalities detected. Immediate medical consultation recommended."
            if severity == "high"
            else "Abnormal patterns detected. Prompt medical evaluation advised."
        )
        return "Malignant", message

# Home Page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# About Page
@app.route("/about")
def about():
    return render_template("about.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for('home'))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for('home'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Preprocess image
        img = preprocess_image(file_path)

        # Perform segmentation
        prediction = segmentation_model.predict(img)[0]

        # Save the result image
        result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result_" + filename)
        tf.keras.preprocessing.image.save_img(result_path, prediction)

        # Analyze prediction
        diagnosis, description = analyze_prediction(prediction)
        
        # Store results in session
        session['diagnosis'] = diagnosis
        session['description'] = description
        session['original_image'] = filename
        session['result_image'] = "result_" + filename

        print(f"Diagnosis: {diagnosis}, Description: {description}")  # Debugging print
        
        return redirect(url_for('results'))

    return redirect(url_for('home'))

# Results Page
@app.route("/results")
def results():
    if 'diagnosis' not in session:
        return redirect(url_for('home'))
        
    return render_template(
        "results.html",
        diagnosis=session['diagnosis'],
        description=session['description'],
        original_image=session['original_image'],
        result_image=session['result_image']
    )

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
