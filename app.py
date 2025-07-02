import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model

app = Flask(__name__)
model = load_model('emotion_model.keras')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    
    if not os.path.exists('static'):
        os.makedirs('static')
    
    image_path = os.path.join('static', 'uploaded.jpg')
    file.save(image_path)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    prediction_label = "No face detected"

    for x, y, w, h in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        prediction = model.predict(roi)
        prediction_label = emotion_labels[np.argmax(prediction)]
        break 

    return render_template('result.html', prediction=prediction_label, user_image='uploaded.jpg')

if __name__ == '__main__':
    app.run(debug=True)
