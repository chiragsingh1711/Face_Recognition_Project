from kivy.app import App
import cv2
import os
import face_recognition
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
import numpy as np


class MainApp(App):
    def build(self):
        self.camera_object = Camera()
        # self.camera_object.resolution = (800, 800)

        # Button Adding Image to Datbase
        self.button_object = Button(text="Add Image")
        self.button_object.size_hint = (0.2, 0.2)
        self.button_object.pos_hint = {'x': 0.25, 'y': 0}
        self.button_object.bind(on_press=self.take_selfie)

        # Adding Indentify the person
        self.button_object2 = Button(text="Identify")
        self.button_object2.size_hint = (0.2, 0.2)
        self.button_object2.pos_hint = {'x': 0.30, 'y': 0}
        self.button_object2.bind(on_press=self.Identify)

        # Text Input
        self.text_object = TextInput(text="Enter name:")
        self.text_object.size_hint = (0.2, 0.2)
        self.text_object.pos_hint = {'x': 0.40, 'y': 0}

        # Submit Button
        self.submit_object = Button(text="Submit")
        self.submit_object.size_hint = (0.2, 0.2)
        self.submit_object.pos_hint = {'x': 0.5, 'y': 0}
        self.submit_object.bind(on_press=self.Submit)

        # Layout
        self.layout_object = BoxLayout()
        self.layout_object.add_widget(self.camera_object)
        self.layout_object.add_widget(self.button_object)
        self.layout_object.add_widget(self.button_object2)
        self.layout_object.add_widget(self.text_object)
        self.layout_object.add_widget(self.submit_object)
        return self.layout_object

    def Submit(self, *args):
        # print(self.text_object.text)
        pass

    def take_selfie(self, *args):
        print("Image Added to Database")

        File_name = "Database/" + "_".join(self.text_object.text.split()) + ".png"
        self.camera_object.export_to_png(File_name)

    def Identify(self, *args):
        file_name = "Identifying.png"
        self.camera_object.export_to_png(file_name)

        def findEncodings(images):
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                if len(encode) > 0:
                    encodeList.append(encode)
                return encodeList

        names = []
        images = []
        myList = []
        path = "Database"
        myList = os.listdir(path)
        # print(myList)

        for c1 in myList:
            CurImg = cv2.imread(f'{path}/{c1}')
            # CurImg = cv2.imread(f'../Database/{c1}')
            images.append(CurImg)
            names.append(os.path.splitext(c1)[0])
        # print(names)
        # print(myList)
        encodelist_known = findEncodings(images)
        # print(len(encodelist_known))

        file_name = "Identifying.png"
        img = face_recognition.load_image_file(file_name)
        imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceloc = face_recognition.face_locations(imgs)
        encode_img = face_recognition.face_encodings(imgs, faceloc)
        # print(encode_img)
        for encodeface, FaceLoc in zip(encode_img, faceloc):
            matches = face_recognition.compare_faces(encodelist_known, encodeface)
            facedis = face_recognition.face_distance(encodelist_known, encodeface)
            matchIndex = np.argmin(facedis)

            if matches[matchIndex]:
                name = names[matchIndex]
                print(name)


if __name__ == '__main__':
    app = MainApp()
    app.run()
