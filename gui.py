import tkinter as tk
from tkinter import filedialog
import time
import cv2
import serial
import numpy as np
import PIL.Image, PIL.ImageTk  # pip install Pillow

gui = tk.Tk()
gui.geometry("500x500")
gui.title("Animal detector")
gui.configure(bg="white")

start = tk.Frame(gui, bg="white")
show = tk.Frame(gui, bg="white")

kamera = tk.BooleanVar()
plik = tk.BooleanVar()
wyswietlanie = tk.BooleanVar()
zapisywanie = tk.BooleanVar()
sygnal = tk.BooleanVar()
nazwa_filmu_we = tk.StringVar()
deviceID = 0;  # 0 = open default camera
apiID = cv2.CAP_ANY;  # 0 = autodetect default API

# Klasy
sarna = tk.BooleanVar()
dzik = tk.BooleanVar()
ptak = tk.BooleanVar()
koza = tk.BooleanVar()
wybrane_klasy = []

# ESP32
ser = serial.Serial()
ser.baudrate = 115200
ser.bytesize = 8
ser.port = 'COM3' # zależy od systemu

# ustawienia
def getFilePath():
    if plik.get() and not kamera.get():
        filename = filedialog.askopenfilename(initialdir="/", title="Wybierz plik", filetypes=(
            ("avi files", "*.avi"), ("mp4 files", "*.mp4"), ("all files", "*.*")))
        nazwa_filmu_we.set(filename)


def showPrediction():
    show.pack(fill='both', expand=1)
    start.pack_forget()
    prep()


l1 = tk.Label(start, text="Wybierz", bg="white")
l1.pack()
c1 = tk.Checkbutton(start, text="Film z kamery", variable=kamera, onvalue=True, offvalue=False, bg="white")
c1.pack()
c2 = tk.Checkbutton(start, text="Film z pliku", variable=plik, onvalue=True, offvalue=False, bg="white")
c2.pack()
b1 = tk.Button(start, text="Wybierz plik wideo", command=getFilePath, bg="white", foreground="black")
b1.pack()
c3 = tk.Checkbutton(start, text="Wyświetlanie", variable=wyswietlanie, onvalue=True, offvalue=False, bg="white")
c3.pack()
c4 = tk.Checkbutton(start, text="Zapisywanie", variable=zapisywanie, onvalue=True, offvalue=False, bg="white")
c4.pack()
c5 = tk.Checkbutton(start, text="Sygnał do płytki", variable=sygnal, onvalue=True, offvalue=False, bg="white")
c5.pack()

l2 = tk.Label(start, text="Wybierz zwierzęta do wykrywania", bg="white")
l2.pack()
c6 = tk.Checkbutton(start, text="Sarna", variable=sarna, onvalue=True, offvalue=False, bg="white")
c6.pack()
c7 = tk.Checkbutton(start, text="Dzik", variable=dzik, onvalue=True, offvalue=False, bg="white")
c7.pack()
c8 = tk.Checkbutton(start, text="Ptak", variable=ptak, onvalue=True, offvalue=False, bg="white")
c8.pack()
c9 = tk.Checkbutton(start, text="Koza", variable=koza, onvalue=True, offvalue=False, bg="white")
c9.pack()

b2 = tk.Button(start, text="Dalej", command=showPrediction, bg="white", foreground="black")
b2.pack()

start.pack()

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


def process_img(img, wykrywane_klasy=None):
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


def prep():
    global video, size, fps, out, canvas, ser,wybrane_klasy
    if plik:
        video = cv2.VideoCapture(nazwa_filmu_we.get())
    else:
        video = cv2.VideoCapture(deviceID, apiID)

    # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = video.get(cv2.CAP_PROP_FPS)
    print(size, fps)
    if zapisywanie:
        out = cv2.VideoWriter('wyjscie.mp4', -1, fps, size)
    if wyswietlanie:
        canvas = tk.Canvas(show, width=size[0], height=size[1])
        gui.geometry(str(size[0]) + 'x' + str(size[1] + 50))
        canvas.pack()
    if sygnal.get():
        ser.open()

    #wybieranie klas
    wybrane_klasy = []
    #if sarna:
    #    wybrane_klasy.append(classes.index("deer")) #- nie ma sarny
    #if dzik :
    #    wybrane_klasy.append(classes.index("boar")) #- nie ma dzika
    if ptak.get():
        wybrane_klasy.append(classes.index("bird"))
    #if koza:
    #    wybrane_klasy.append(classes.index("goat")) #- nie ma kozy

    if len(wybrane_klasy) == 0:
        wybrane_klasy = None

    print("wybrane klasy: ", wybrane_klasy)

def detect():
    global photo, ser, wybrane_klasy
    if run:
        ret, frame = video.read()
        if ret:

            # TODO: przetwarzanie obrazu

            klasy = process_img(frame, wybrane_klasy)
            klasy_nazwy = [classes[k] for k in klasy]
            print(klasy_nazwy)

            # ############################
            if zapisywanie:
                out.write(frame)

            if wyswietlanie:
                photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                canvas.create_image((0, 0), image=photo, anchor='nw')
                canvas.update()

            if sygnal:
                if len(klasy) > 0:
                    ser.write(b'1')
                else:
                    ser.write(b'0')
        else:
            video.release()
            if zapisywanie:
                out.release()
    show.after(1, detect)

run = False
def startFilm():
    global run
    run = True

def stopFilm():
    global run
    run = False


b3 = tk.Button(show, text="START", command=startFilm, bg="white", foreground="black")
b3.pack(side=tk.TOP)
b4 = tk.Button(show, text="STOP", command=stopFilm, bg="white", foreground="black")
b4.pack(side=tk.TOP)
show.after(1, detect)

gui.mainloop()
