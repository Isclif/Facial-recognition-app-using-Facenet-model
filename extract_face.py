import cv2
import os
from keras_facenet import FaceNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



'''
img = cv2.imread('moi.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.imshow('moi', img)
# plt.imshow(img)
# plt.show()

detector = MTCNN()
resultat = detector.detect_faces(img)

print(resultat)

x,y,w,h = resultat[0]['box']

img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 30)
# plt.imshow(img)
# plt.show()

ma_face = img[y:y+h, x:x+w]
ma_face = cv2.resize(ma_face, (160,160))
plt.imshow(ma_face)
plt.show()

print(ma_face)
'''

class faceloading:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (self.target_size))

        return face_resize

    def load_faces(self, dir):
        faces = []
        for img_name in os.listdir(dir):
            try:
                path = dir + img_name
                single_face = self.extract_face(path)
                faces.append(single_face)
            except Exception as e:
                pass
        return faces

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory + '/' + sub_dir + '/'
            faces = self.load_faces(path)
            labels = [sub_dir for _ in range(len(faces))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(faces)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)



load = faceloading('images/train')
X,Y = load.load_classes()

# load.plot_image()

model = FaceNet()
print('Facenet loaded')


def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis = 0) #4D (Nonex160x160x3)
    yhat = model.embeddings(face_img)
    return yhat[0]

embedded_x = []

for img in X :
    embedded_x.append(get_embedding(img))

embedded_x = np.asarray(embedded_x)

np.savez_compressed('faces_embeddings.npz', embedded_x, Y)

print('NPZ created and saved successfully')