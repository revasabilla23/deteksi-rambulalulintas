# lokal akses

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

model = load_model("mobilenetv2_final_traffic-sign.h5")

with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f]

IMG_SIZE = 224

cap = cv2.VideoCapture(0)
print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = img_to_array(image_rgb) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)
    pred_index = np.argmax(predictions)
    pred_label = class_names[pred_index]
    confidence = np.max(predictions) * 100

    label = f"{pred_label} ({confidence:.1f}%)"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Deteksi Rambu", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
