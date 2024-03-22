import cv2
import numpy as np
from PIL import Image
import os

# Membuat recognizer dan detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Fungsi untuk memuat data latih
def getImagesWithLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []
    
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        # Deteksi wajah dalam gambar
        faces = detector.detectMultiScale(imageNp, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Preprocessing gambar
            faceSamples.append(cv2.equalizeHist(imageNp[y:y+h, x:x+w]))
            Ids.append(Id)
    
    return faceSamples, Ids

# Memuat data latih
faces, Ids = getImagesWithLabels('Dataset')

# Melatih recognizer dengan data latih
recognizer.train(faces, np.array(Ids))

# Menyimpan model yang dilatih
recognizer.save('Dataset/training.xml')
