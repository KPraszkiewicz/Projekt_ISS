import sys
import time
import cv2
import numpy as np
import serial
from PyQt6.QtGui import QAction, QPixmap, QColor, QImage
from PyQt6.QtCore import QThread, pyqtSignal, QSettings, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QVBoxLayout,
    QCheckBox,
    QGridLayout,
    QRadioButton,
    QFileDialog,
    QPushButton,
    QMainWindow,
    QToolBar,
    QLabel,
    QWidget,
    QHBoxLayout,

)

# Set camera
deviceID = 0;  # 0 = open default camera
apiID = cv2.CAP_ANY;  # 0 = autodetect default API

# ESP32
ser = serial.Serial()
ser.baudrate = 115200
ser.bytesize = 8
ser.port = 'COM3' # zależy od systemu

# Load names of classes and get random colors
WHITE = (255, 255, 255)
classes = open('coco.names').read().strip().split('\n')
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# determine the output layer
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

conf = 0.5


class Options(QDialog):
    submitClicked = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Options")
        dialogLayout = QVBoxLayout()
        gridLayout = QGridLayout()

        self.kamera = QRadioButton("Wczytaj film z kamery")
        self.kamera.setChecked(True)
        gridLayout.addWidget(self.kamera)
        self.plik = QRadioButton("Wczytaj film z pliku")
        gridLayout.addWidget(self.plik)

        self.filename = QLineEdit()
        self.filename.setObjectName("filepath")
        self.filename.setReadOnly(True)

        self.getFileBtn = QPushButton("Wybierz plik")
        self.getFileBtn.clicked.connect(self.browse_files)
        gridLayout.addWidget(self.getFileBtn)

        self.wyswietlanie = QCheckBox("Wyświetlanie")
        gridLayout.addWidget(self.wyswietlanie)
        self.zapisywanie = QCheckBox("Zapisywanie")
        gridLayout.addWidget(self.zapisywanie)
        self.sygnal = QCheckBox("Sygnał do płytki")
        gridLayout.addWidget(self.sygnal)

        dialogLayout.addLayout(gridLayout)
        buttons = QDialogButtonBox()
        buttons.setStandardButtons(
            QDialogButtonBox.StandardButton.Save
        )
        buttons.accepted.connect(self.accept)

        dialogLayout.addWidget(buttons)
        self.setLayout(dialogLayout)

    def browse_files(self):
        if not self.plik.isChecked():
            return
        filename = QFileDialog.getOpenFileName(self, "Wybierz plik", '',
                                               "Avi files (*.avi);; Mp4 files (*.mp4);; All files (*)")
        if filename:
            self.filename.setText(filename[0])

    def accept(self):
        ret = {'kamera': self.kamera.isChecked(), 'plik': self.plik.isChecked(), 'filepath': self.filename.text(),
               'wyswietlanie': self.wyswietlanie.isChecked(), 'zapisywanie': self.zapisywanie.isChecked(),
               'sygnal': self.sygnal.isChecked()}
        self.submitClicked.emit(ret)
        self.close()


class Classes(QDialog):
    submitClicked = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Class Options")
        dialogLayout = QVBoxLayout()
        gridLayout = QGridLayout()

        self.cat = QCheckBox("Kot")
        self.cat.setChecked(True)
        gridLayout.addWidget(self.cat)

        self.dog = QCheckBox("Pies")
        self.dog.setChecked(True)
        gridLayout.addWidget(self.dog)

        self.bird = QCheckBox("Ptak")
        self.bird.setChecked(True)
        gridLayout.addWidget(self.bird)

        self.cow = QCheckBox("Krowa")
        self.cow.setChecked(True)
        gridLayout.addWidget(self.cow)

        self.horse = QCheckBox("Koń")
        self.horse.setChecked(True)
        gridLayout.addWidget(self.horse)

        self.sheep = QCheckBox("Owca")
        self.sheep.setChecked(True)
        gridLayout.addWidget(self.sheep)

        dialogLayout.addLayout(gridLayout)
        buttons = QDialogButtonBox()
        buttons.setStandardButtons(
            QDialogButtonBox.StandardButton.Save
        )
        buttons.accepted.connect(self.accept)

        dialogLayout.addWidget(buttons)
        self.setLayout(dialogLayout)

    def accept(self):
        ret = {'cat': self.cat.isChecked(), 'dog': self.dog.isChecked(), 'bird': self.bird.isChecked(),
               'cow': self.cow.isChecked(), 'horse': self.horse.isChecked(), 'sheep': self.sheep.isChecked()}
        self.submitClicked.emit(ret)
        self.close()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, video, out, fps, size, wybrane_klasy, wyswietlanie, zapisywanie, sygnal, parent=None):
        QThread.__init__(self, parent)
        self._run_flag = True
        self.running = False
        self.video = video
        self.out = out
        self.fps = fps
        self.size = size
        self.wybrane_klasy = wybrane_klasy
        self.wyswietlanie = wyswietlanie
        self.zapisywanie = zapisywanie
        self.sygnal = sygnal

    def run(self):
        while self._run_flag:
            if not self.running:
                continue
            ret, frame = self.video.read()
            if ret:
                klasy = self.process_img(img=frame, wykrywane_klasy=self.wybrane_klasy)
                klasy_nazwy = [classes[k] for k in klasy]
                print(klasy_nazwy)

                # ############################
                if self.zapisywanie:
                    self.out.write(frame)

                if self.wyswietlanie:
                    self.change_pixmap_signal.emit(frame)
                if self.sygnal:
                    if len(klasy) > 0:
                        ser.write(b'1')
                    else:
                        ser.write(b'0')
            else:
                self.video.release()
                if self.zapisywanie:
                    self.out.release()

    def end(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    def startDetection(self):
        self.running = True

    def stopDetection(self):
        self.running = False

    def process_img(self, img, wykrywane_klasy=None):
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        t0 = time.time()
        outputs = net.forward(ln)
        t = time.time() - t0

        print(t)
        # combine the 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)

        H, W = img.shape[:2]
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf and (not wykrywane_klasy or classID in wykrywane_klasy):
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w // 2), int(y - h // 2)
                p1 = int(x + w // 2), int(y + h // 2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                # cv.rectangle(img, p0, p1, WHITE, 1)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf - 0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return classIDs


class Player(QWidget):
    def __init__(self, video, out, fps, size, wybrane_klasy, wyswietlanie, zapisywanie, sygnal):
        super(Player, self).__init__()

        self.w = size[0]
        self.h = size[1]

        mainLayout = QVBoxLayout(self)
        boxLayout = QHBoxLayout(self)
        boxLayout

        startBtn = QPushButton("Start")
        startBtn.clicked.connect(self.onStart)
        boxLayout.addWidget(startBtn)

        stopBtn = QPushButton("Stop")
        stopBtn.clicked.connect(self.onStop)
        boxLayout.addWidget(stopBtn)

        self.videoPlayer = QLabel()
        self.videoPlayer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gray = QPixmap(self.w, self.h)
        gray.fill(QColor("darkGray"))
        self.videoPlayer.setPixmap(gray)
        mainLayout.addWidget(self.videoPlayer)

        mainLayout.addLayout(boxLayout)
        self.setLayout(mainLayout)

        self.thread = VideoThread(video, out, fps, size, wybrane_klasy, wyswietlanie, zapisywanie, sygnal)
        self.thread.change_pixmap_signal.connect(self.updateImage)
        self.thread.start()

    def updateImage(self, cvImg):
        qtImg = self.convert_cv_qt(cvImg)
        self.videoPlayer.setPixmap(qtImg)

    def convert_cv_qt(self, cv_img):
        # Convert from an opencv image to QPixmap
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.w, self.h, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        self.thread.end()
        event.accept()

    def onStart(self):
        self.thread.startDetection()

    def onStop(self):
        self.thread.stopDetection()


class Detector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.options = None
        self.classes = None

        # opcje
        self.kamera = True
        self.plik = False
        self.filename = ''
        self.wyswietlanie = False
        self.zapisywanie = False
        self.sygnal = False

        # klasy
        self.cat = True
        self.dog = True
        self.bird = True
        self.cow = True
        self.horse = True
        self.sheep = True

        # przetwarzanie
        self.video = None
        self.size = None
        self.fps = None
        self.wybrane_klasy = None
        self.player = None
        self.out = None

        self.setWindowTitle("Animal Detector")
        self.setGeometry(0, 0, 1280, 720)

        optionsBtn = QAction("Detection Options", self)
        optionsBtn.triggered.connect(self.optionsClicked)

        classesBtn = QAction("Detection Classes", self)
        classesBtn.triggered.connect(self.classesClicked)

        toolbar = QToolBar()
        toolbar.addAction(classesBtn)
        toolbar.addAction(optionsBtn)
        self.addToolBar(toolbar)

    def optionsClicked(self):
        self.options = Options()
        self.options.submitClicked.connect(self.onOptionsConfirmed)
        self.options.show()

    def onOptionsConfirmed(self, choices):
        self.kamera = choices['kamera']
        self.plik = choices['plik']
        self.filename = choices['filepath']
        self.wyswietlanie = choices['wyswietlanie']
        self.zapisywanie = choices['zapisywanie']
        self.sygnal = choices['sygnal']
        print(choices)
        self.prepPlayer()

    def classesClicked(self):
        self.classes = Classes()
        self.classes.submitClicked.connect(self.onClassesConfirmed)
        self.classes.show()

    def onClassesConfirmed(self, choices):
        self.cat = choices['cat']
        self.dog = choices['dog']
        self.bird = choices['bird']
        self.cow = choices['cow']
        self.horse = choices['horse']
        self.sheep = choices['sheep']
        print(choices)
        self.prepClasses()

    def prepClasses(self):
        # wybieranie klas
        self.wybrane_klasy = []
        if self.cat:
            self.wybrane_klasy.append(classes.index("cat"))
        if self.dog:
            self.wybrane_klasy.append(classes.index("dog"))
        if self.bird:
            self.wybrane_klasy.append(classes.index("bird"))
        if self.horse:
            self.wybrane_klasy.append(classes.index("horse"))
        if self.cow:
            self.wybrane_klasy.append(classes.index("cow"))
        if self.sheep:
            self.wybrane_klasy.append(classes.index("sheep"))

        if len(self.wybrane_klasy) == 0:
            self.wybrane_klasy = None

        print("wybrane klasy: ", self.wybrane_klasy)

    def prepPlayer(self):
        if self.plik:
            self.video = cv2.VideoCapture(self.filename)
        else:
            self.video = cv2.VideoCapture(deviceID, apiID)

        # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        self.size = (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        print(self.size, self.fps)
        if self.zapisywanie:
            self.out = cv2.VideoWriter('wyjscie.mp4', -1, self.fps, self.size)
        if self.sygnal:
            ser.open()
        self.player = Player(self.video, self.out, self.fps, self.size, self.wybrane_klasy, self.wyswietlanie,
                             self.zapisywanie, self.sygnal)
        self.setCentralWidget(self.player)


if __name__ == "__main__":
    app = QApplication([])
    window = Detector()
    window.show()
    sys.exit(app.exec())
