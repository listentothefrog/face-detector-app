import cv2

img_file = "lady_smiling.png"

img = cv2.imread(img_file)

# classifier files 👇

eye_classifier = "haarcascade_eye.xml"
face_classifier = "haarcascade_frontalface_default.xml"
smile_classifier = "haarcascade_smile.xml"

eye_tracker = cv2.CascadeClassifier(eye_classifier)
face_tracker = cv2.CascadeClassifier(face_classifier)
smile_tracker = cv2.CascadeClassifier(smile_classifier)

# change the colour to black and white 👇
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detecting face 👇
faces = face_tracker.detectMultiScale(gray_img, scaleFactor=1.7, minNeighbors=5)
eyes = eye_tracker.detectMultiScale(gray_img, minNeighbors=10)
smiles = smile_tracker.detectMultiScale(gray_img, scaleFactor=1.7, minNeighbors=30)
print(faces)
print(eyes)
print(smiles)

# draws around the object 👇
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
for (x, y, w, h) in smiles:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

# display the image in a window 👇
cv2.imshow('Face Detector', img)
cv2.waitKey()
print('Code Completed!')
