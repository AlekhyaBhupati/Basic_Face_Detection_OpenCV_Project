import cv2
import numpy as np
import face_recognition

imgOriginal = face_recognition.load_image_file('data/ben_afflek.jpg')
imgOriginal = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB)
imgOriginal = cv2.resize(imgOriginal, (500,500))

imgTest = face_recognition.load_image_file('data/ben_afflek_test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
imgTest = cv2.resize(imgTest, (500,500))
#imgTest = cv2.rotate(imgTest, cv2.ROTATE_90_CLOCKWISE)

faceloc = face_recognition.face_locations(imgOriginal)[0]
encodealekhya = face_recognition.face_encodings(imgOriginal)[0]
cv2.rectangle(imgOriginal,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodealekhya],encodeTest)
faceDis = face_recognition.face_distance([encodealekhya],encodeTest)
print(results, faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Original_image', imgOriginal)
cv2.imshow('Test_image', imgTest)
cv2.waitKey(0)

