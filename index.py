import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import dlib
from tensorflow.keras.models import Sequential, load_model, Model

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.image = ''
        self.cameraimage = 'null'
        # Создание кнопок
        self.load_button = QPushButton("Загрузить изображение")
        self.photo_button = QPushButton("Сфотографировать")
        self.evaluate_button = QPushButton("Оценить функциональное состояние")
        self.evaluate_button.setEnabled(False)

        # Создание метки для отображения изображения
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # Создание метки для отображения результата оценки
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)

        # Создание вертикального и горизонтального макетов
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()

        # Добавление кнопок в горизонтальный макет
        hbox.addWidget(self.load_button)
        hbox.addWidget(self.photo_button)
        hbox.addWidget(self.evaluate_button)

        # Добавление меток в вертикальный макет
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.result_label)

        # Добавление горизонтального макета в вертикальный макет
        vbox.addLayout(hbox)

        # Установка вертикального макета в качестве макета окна
        self.setLayout(vbox)

        # Подключение функций-обработчиков к кнопкам
        self.load_button.clicked.connect(self.load_image)
        self.photo_button.clicked.connect(self.take_photo)
        self.evaluate_button.clicked.connect(self.evaluate_image)

    def load_image(self):
        # Открытие диалогового окна для выбора файла
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.png *.xpm *.jpg *.bmp)")

        if file_path:
            # Загрузка изображения
            img = cv2.imread(file_path)
            self.image = file_path
            self.cameraimage = 'null'
            '''# Преобразование изображения в QPixmap и отображение на метке
            # pixmap = QPixmap.fromImage(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap)'''
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaled(720, 405, QtCore.Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

            # Активация кнопки "оценить функциональное состояние"
            self.evaluate_button.setEnabled(True)


    def take_photo(self):
        # Создание объекта VideoCapture для захвата видео с камеры
        cap = cv2.VideoCapture(0)

        # Захват кадра с камеры
        ret, frame = cap.read()

        # Остановка захвата видео
        cap.release()

        # Сохранение кадра в файл
        # cv2.imwrite("photo.jpg", frame)

        # Отображение кадра на экране
        # cv2.imshow("Photo", frame)

        self.cameraimage = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(720, 405, QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

        # Активация кнопки "оценить функциональное состояние"
        self.evaluate_button.setEnabled(True)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    def evaluate_image(self):
        # Получение класса изображения
        self.predict_image()


    def predict_image(self):

        classes_name = ['бодрый', 'устал - иди отдохни, или скажи мяу']

        path = self.image

        imagea = cv2.imread(path)

        if(self.cameraimage != 'null'):
            imagea = self.cameraimage

        gray = cv2.cvtColor(imagea, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            tl_x = faces[0].tl_corner().x
            tl_y = faces[0].tl_corner().y
            tl_h = faces[0].height()
            tl_w = faces[0].width()
            if tl_x > 0 and tl_y > 0 and tl_h > 10 and tl_w > 10:
                imagea = imagea[tl_y: tl_y + tl_w, tl_x:tl_x + tl_h, :]
        image_norm = imagea / 255.0
        im = cv2.resize(image_norm, (48, 48))
        prediction = model.predict(im[np.newaxis])
        index = np.argmax(prediction)

        # Отображение результата оценки на метке
        self.result_label.setText(f"Класс: {str(classes_name[index])}")


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()

    # Создание объекта модели ResNet50
    model = load_model("fatigue_model4.model")

    # Создание объекта приложения PyQt
    app = QApplication(sys.argv)

    # Создание объекта главного окна
    window = MainWindow()

    # Отображение главного окна
    window.show()

    # Запуск главного цикла приложения
    sys.exit(app.exec_())

