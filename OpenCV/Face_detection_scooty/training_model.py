import cv2
import os
import numpy as np
from PIL import Image
import requests

def assure_path_exists(path):
    # Same as in face_dataset.py
    if not os.path.exists(path):
        os.makedirs(path)

# Initialize LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to get images and labels
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples, ids = [], []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

faces, ids = getImagesAndLabels('dataset')
recognizer.train(faces, np.array(ids))

# Save the model into trainer.yml
assure_path_exists('trainer/')
recognizer.save('trainer/trainer.yml')

# Send trained model to the backend API
url = 'http://your-backend-api-url/train'
files = {'file': open('trainer/trainer.yml', 'rb')}
response = requests.post(url, files=files)

print(response.json())
