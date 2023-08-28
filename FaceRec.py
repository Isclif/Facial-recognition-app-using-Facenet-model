import customtkinter
from PIL import ImageTk
import imutils
import cv2
import numpy as np
import os
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pickle
from keras_facenet import FaceNet
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings
# from test import test


# variables

cap = None
frame = None
toplevel = None
intru = 0

model_facenet = FaceNet()
faces_embedding = np.load('faces_embeddings.npz')

embedded_x, Y = faces_embedding['arr_1'], faces_embedding['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = pickle.load(open('SVM_model_160x160.npz', 'rb'))

class FaceRecApp(customtkinter.CTk):
    width = 800
    height = 600

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()
        self.title('Reconaissance faciale en temps réel')
        self.geometry('%dx%d' % (self.width, self.height))
        self.state('zoomed')
        self.resizable(False,False)
        self.grab_set()
        # self.iconphoto(False, ImageTk.PhotoImage(file='customtkinter_icon_windows.ico'))


        # Changes Theme
        self.appearance_mode = customtkinter.CTkOptionMenu(self, values=["Dark", "Light", "System"],
                                                           command=self.change_appearance_mode_event)
        self.appearance_mode.place(x=1365, y=30)

        # button change service
        self.change_service = customtkinter.CTkButton(self, text="Changer de mode", command=self.changer_mode)
        self.change_service.place(x=30, y=30)

        self.changer_mode = customtkinter.CTkLabel(self, text="Application de Reconnaissance faciale", font=customtkinter.CTkFont(size=40, weight='bold'))
        self.changer_mode.place(x=380, y=10)

        # Frame, label playing
        self.frame_playing = customtkinter.CTkFrame(self)
        self.frame_playing.pack(padx=(270), pady=(90), fill="both", expand=True)
        self.lancer_camera1 = customtkinter.CTkLabel(self.frame_playing, text="", font=customtkinter.CTkFont(size=40, weight='bold'))
        self.lancer_camera1.pack()

        self.button_lancer_camera1 = customtkinter.CTkButton(self, text='lancer la camera', command=self.start)
        self.button_lancer_camera1.place(x=600, y=790)
        self.arreter_camera = customtkinter.CTkButton(self, text='stop camera', fg_color = ("red", "darkred"), command=self.stop_camera)
        self.arreter_camera.place(x=800, y=790)

        self.f()

    # functions to call with buttons

    def marquer_la_presence(self, name):
        global frame
        with open('presence.csv', 'r+') as f:
            data_list = f.readlines()
            name_list = []
            for line in data_list:
                entry = line.split(',')
                name_list.append(entry[0])

            jour = datetime.now().strftime('%d/%m/%Y')
            verif = str(name) + '_' + str(jour)

            if verif not in name_list:
                times = datetime.now().strftime('%H-%M-%S')
                time_csv = datetime.now().strftime('%H:%M:%S')
                f.writelines(f'\n{name}_{jour}, {time_csv}')

                cv2.imwrite('Captures/correct/' + name + times + '.png', frame)



    def draw_border(self, img, pt1, pt2, color, thickness, r, d):
        x1, y1 = pt1
        x2, y2 = pt2
        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    def streaming(self):
        global cap, frame, fake, intru
        ret, frame = cap.read()
        if ret :
            frame = cv2.resize(frame, (1230, 840)) # 1275, 880
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            today = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            position = (10, 30)
            date = cv2.putText(frame, today, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            faces = haarcascade.detectMultiScale(frame, 1.3 , 5, minSize=(50, 50))  # ou 1.1 , 4 ou encore 1.3 , 5
            # label = test(image=frame, model_dir='Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models',device_id=0)

            for x, y, w, h in faces:
                img = frame[y:y + h, x:x + w]
                img = cv2.resize(img, (160, 160))
                img = np.expand_dims(img, axis=0)
                ypred = model_facenet.embeddings(img)
                face_name = model.predict(ypred)
                class_index = face_name[0]
                ypred_proba = model.predict_proba(ypred)
                ypred_proba = ypred_proba[0, class_index] * 100

                if (ypred_proba > 70):
                    print(ypred_proba)
                    final_name = encoder.inverse_transform(face_name)[0]
                    self.draw_border(frame, (x, y), (x + w, y + h), (7, 212, 14), 5, 15, 10)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (7,212,14), 2)
                    cv2.putText(frame, str(final_name), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (7, 212, 14), 3,
                                cv2.LINE_AA)
                    # cv2.imwrite('Captures/vrai_personne_de_entreprise/' + str(final_name) + str(vrai) + '.png', frame)
                    self.marquer_la_presence(final_name)

                else:
                    print(ypred_proba)
                    self.draw_border(frame, (x, y), (x + w, y + h), (248, 0, 0), 5, 15, 10)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, 'Inconnu', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (248, 0, 0), 3, cv2.LINE_AA)
                    cv2.imwrite('Captures/intru/' + 'inconnu' + str(intru) + '.png', frame)
                    intru += 1

            img = Image.fromarray(frame)
            ImgTks = ImageTk.PhotoImage(image=img)
            self.lancer_camera1.configure(image=ImgTks)
            self.lancer_camera1.imgtk = ImgTks
            self.after(5, self.streaming)
        else:
            self.lancer_camera1.imgtk = ""
            cap.release()

    def start(self):
        global cap
        cap = cv2.VideoCapture(0)
        self.streaming()

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def stop_camera(self):
        global cap
        cap.release()

    def f(self):
        warnings.filterwarnings('ignore')

    def changer_mode(self):
        global toplevel

        toplevel = customtkinter.CTkToplevel(self)
        toplevel.geometry("400x200+800+300")
        toplevel.title('Changer de mode')
        toplevel.grab_set()

        corps = customtkinter.CTkFrame(toplevel)
        corps.pack(padx=(10), pady=(10), fill="both", expand=True)
        corps_label = customtkinter.CTkLabel(corps, text="aller au mode reconnaissance \nfaciale et de fausse identité ?", font=customtkinter.CTkFont(size=20, weight='bold'))
        corps_label.pack(pady=15)

        oui = customtkinter.CTkButton(corps, text='oui', command= self.rec_fake_face)
        non = customtkinter.CTkButton(corps, text='non', fg_color=('red', "darkred"), command=self.exit)

        oui.place(x=30, y=100)
        non.place(x=200, y=100)

    def rec_fake_face(self):
        global toplevel

        toplevel.withdraw()
        toplevel.destroy()

        self.withdraw()
        os.system('python progress_recFake_bar.py')
        self.destroy()

    def exit(self):
        global toplevel

        toplevel.withdraw()
        toplevel.destroy()


if __name__ == "__main__":
    customtkinter.set_appearance_mode('dark')
    app = FaceRecApp()
    app.mainloop()