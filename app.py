#Website with flask

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model dan label
model = load_model("mobilenetv2_final_traffic-sign.h5")
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f]

def predict_image(path):
    img = load_img(path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    index = np.argmax(pred)
    label = class_names[index]
    confidence = float(np.max(pred)) * 100
    return label, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    label, confidence, image_path = None, None, None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            label, confidence = predict_image(filepath)
            return render_template('index.html', label=label, confidence=confidence, image_path=filepath)
    return render_template('index.html')

@app.route('/predict-frame', methods=['POST'])
def predict_frame():
    file = request.files['frame']
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'frame.jpg')
        file.save(path)
        label, conf = predict_image(path)
        return jsonify({'label': label, 'confidence': f'{conf:.2f}%'})
    return jsonify({'error': 'No frame received'})

if __name__ == '__main__':
    app.run(debug=True)
