from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FileField
from wtforms.validators import DataRequired
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key for production
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Directory for uploaded images
app.config['PREDICTION_FOLDER'] = 'static/predictions'  # Directory for predicted masks
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}  # Allowed image formats

# Load the trained TensorFlow model
model = tf.keras.models.load_model('C:\\Users\\LENOVO\\Desktop\\Land Usage AI\\unet_model.h5')

# Define form for secure file upload with CSRF protection
class UploadForm(FlaskForm):
    image = FileField('image', validators=[DataRequired()])

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for the homepage (upload page)
@app.route('/')
def home():
    form = UploadForm()
    return render_template('upload.html', form=form)

# Route for handling image uploads and predictions
@app.route('/upload', methods=['POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        file = request.files['image']
        if file and allowed_file(file.filename):
            # Securely save the uploaded image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the image for model input
            image = Image.open(filepath).resize((64, 64))  # Resize to match model input (64x64)
            image_array = np.array(image) / 255.0  # Normalize to [0, 1]
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Make prediction with the model
            prediction = model.predict(image_array)
            mask = np.argmax(prediction[0], axis=-1)  # Convert to class indices (for segmentation)

            # Save the predicted mask as an image
            mask_image = Image.fromarray((mask * 50).astype(np.uint8))  # Scale for visualization
            mask_filename = f'mask_{filename}.png'
            mask_filepath = os.path.join(app.config['PREDICTION_FOLDER'], mask_filename)
            mask_image.save(mask_filepath)

            # Calculate class distribution
            classes = ['urban', 'water', 'forest', 'agriculture']  # Adjust based on your model
            class_counts = np.bincount(mask.flatten(), minlength=len(classes))
            class_dist = {cls: int(count) for cls, count in zip(classes, class_counts)}
            most_common_class = classes[np.argmax(class_counts)]

            # Render results page with prediction data
            return render_template('result.html', 
                                 most_common_class=most_common_class,
                                 class_dist=class_dist,
                                 mask_image=mask_filename)
        else:
            # Handle invalid file format
            return render_template('upload.html', form=form, error='Invalid file format. Please upload a PNG, JPG, or JPEG image.')
    # Handle form validation errors
    return render_template('upload.html', form=form, error='Please select a file.')

# Ensure static directories exist and run the app
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PREDICTION_FOLDER'], exist_ok=True)
    app.run(debug=True)