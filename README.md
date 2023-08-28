# Facial-recognition-app-using-Facenet
Fcaial recognition application using Facenet / Application de reconnaissance faciale utilisant Facenet
![ok](https://github.com/Isclif/Facial-recognition-app-using-Facenet/assets/103781091/66bec1da-1f3e-4501-8d8c-5a56c61c5496)

## ENGLISH

This project is a facial recognition application that uses several models which are: Mtcnn model, Facenet model, SVM model, haarCascade model, its operation is detailed as follows:
1- The mtcnn model is used to detect the faces on the images used for training and which will then be transformed into vectors of incorporation by the facenet model
2- The Facenet model creates the embeddings of the faces detected by the mtcnn model (this allows us to train the Facenet model on our own images) and then the embeddings are saved in a file called ''faces_embeddings.npz'' which will constitute our data set.
3- The svm model which allows the classification of the incorporations of the faces that are in our dataset into vector groups that have similar incorporations and saves them in a file called “SVM_model_160x160.npz”.
4- The haarCascade model is used to perform real-time detection of faces present on the camera's video stream; this model will detect the faces on the video stream, then will transmit it to the facenet model which will in turn create an embedding vector for each face present on the video stream and then the embedding vector for each face created will be compare to the vector groups of incorporations used for training which are saved in the file ''SVM_model_160x160.npz'' and the name of the vector group to which it belongs will be returned to the screen; Otherwise it will return “unknown”.

### Install dependency Library
```
pip install -r requirements.txt
```
### Clone
```
git clone https://github.com/Isclif/Facial-recognition-app-using-Facenet
```
### How to use the App ?
```
Run the extract_face program to train the Facenet model
Then launch the program '''SVM_model'' for the classification of vector incorporations into groups of similar vectors
And finally launch “Facerec” for and press the button “Launch the camera” to test the facial recognition system
```
If you have any problems regarding this project contact me.

## Enjoy the app.


## FRENCH

Ce projet est une application de reconnaissance faciale qui utilise plusieurs modèles qui sont : Mtcnn modèle, Facenet modèle, SVM(Support Vector Machines), haarCascade modèle, son fonctionnement est détaillé comme suit : 
1-	Le mtcnn modèle est utiliser pour détecter les faces sur les images utiliser pour l’entrainement et qui seront par la suite transformer en vecteurs dincorporation par le modèle facenet
2-	Le modèle Facenet cree les incorporations des faces détecter par le modèle mtcnn (ceci nous permet d’entrainer le modèle Facenet sur nos propres images) et ensuite les incorporations sont enregistrées dans un fichier appeler ‘’faces_embeddings.npz’’ qui constituera notre dataset.
3-	Le svm(Support Vector Machines) modèle qui permet la classification des incorporations des faces qui se trouves dans notre dataset en groupes de vecteur qui ont des incorporations similaires et les enregistre dans un fichier appeler ‘’SVM_model_160x160.npz’’.
4-	Le haarCascade modèle est utiliser pour effectuer la détection en temps réelle des visages présents sur le flux vidéo de la camera ; ce modèle va détecter les visages sur le flux vidéo, ensuite va le transmettre au modèle facenet qui va à son tour créer un vecteur d’incorporation pour chaque visage présent sur le flux vidéo et ensuite le vecteur d’incorporation de chaque face créer va être comparer aux groupes de vecteur d’incorporations utiliser pour l’entrainement qui sont enregistrer dans le fichier ‘’SVM_model_160x160.npz’’ et le nom du groupe de vecteur auquel il appartient sera renvoyer à l’écran ; Dans le cas contraire il va renvoyer ‘’inconnu’’. 

### Install dependency Library  
```
pip install -r requirements.txt
```
### Clone
```
git clone https://github.com/Isclif/Facial-recognition-app-using-Facenet
``` 
### Comment utiliser l'application ?
```
Lancez le programme ‘’extract_face’’ pour l’entrainement du modèle Facenet
Ensuite lancez le programme ‘’’SVM_model’’ pour la classification des vecteur d’incorporations en groupe de vecteurs similaires
Et enfin lancez ‘’Facerec’’ pour et appuyez sur le buton ‘’Lancer la camera’’ pour tester le système de reconnaissance faciale
```

Si vous avez des problemes concernant ce project contactez moi.

## Profitez de l’application.


