from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from numpy import load
import cv2
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import numpy as np
import pickle

embedding_images = load('faces_embeddings.npz')
embedded_x, Y = embedding_images['arr_0'], embedding_images['arr_1']

encoder = LabelEncoder()
encoder.fit(Y)

Y = encoder.transform(Y)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(embedded_x, Y, shuffle=True, random_state=17)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)
pickle.dump(model, open('SVM_model_160x160.npz', 'wb'))
ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)

acc1 = accuracy_score(Y_train, ypreds_train)

acc2 = accuracy_score(Y_test, ypreds_test)

# print(acc1)
# print(acc2)

# test
facenet_model = FaceNet()

def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis = 0) #4D (Nonex160x160x3)
    yhat = facenet_model.embeddings(face_img)
    return yhat[0]

print('getting face...')
my_image = cv2.imread('test.jpg')
my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)
detector = MTCNN()
x,y,w,h = detector.detect_faces(my_image)[0]['box']
my_image = my_image[y:y+h, x:x+w]
my_image = cv2.resize(my_image, (160,160))
print('laoding embedding...')
test_image = get_embedding(my_image)

test_image = [test_image]
ypreds_prob = model.predict_proba(test_image)
ypreds = model.predict(test_image)
# print(ypreds_prob)
# print(ypreds)
class_index = ypreds[0]
class_probability = ypreds_prob[0,class_index] * 100
predict_name = encoder.inverse_transform(ypreds)
print('Predicted: %s (%.3f)' % (predict_name[0], class_probability))



