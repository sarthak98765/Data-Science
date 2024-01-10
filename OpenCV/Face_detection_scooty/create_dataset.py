import cv2
import requests
import json

def check_path(path):
    # Function to confirm whether the given path exists or not
    if not os.path.exists(path):
        os.makedirs(path)

# Start video capturing
vid_cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = 1
count = 0

check_path("dataset/")

while True:
    _, image_frame = vid_cam.read()

    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.4, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        count += 1
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])

        cv2.imshow('Creating Dataset!!!', image_frame)

        # Break out of the loop after capturing the first face
        break

    if cv2.waitKey(1) == ord('q') or count > 0:
        break

vid_cam.release()
cv2.destroyAllWindows()

# Send data to the backend API
url = 'http://your-backend-api-url/detect'
files = {'file': open('dataset/User.{face_id}.{count}.jpg', 'rb')}
response = requests.post(url, files=files)

print(response.json())
