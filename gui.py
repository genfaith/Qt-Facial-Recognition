import cv2 as cv
import sys

from PySide6.QtWidgets import QWidget, QLabel, QApplication, QPushButton
from PySide6.QtCore import QThread, Qt, Signal as qtsignal, Slot as qtslot, QRect
from PySide6.QtGui import QImage, QPixmap, QCursor, QIcon
import PySide6.QtCore as QtCore

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(select_largest=False, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load embeddings
saved_embedding_data = torch.load('data.pt')
embedding_list = saved_embedding_data[0]
name_list = saved_embedding_data[1]


def recognize_face(face_img):

  print(face_img.shape)
  embedding = resnet(face_img.unsqueeze(0)).detach().to(device)
  embedding_dist_list = []

  for idx, embedding_from_db in enumerate(embedding_list):
    embedding_dist = torch.dist(embedding, embedding_from_db).item()
    embedding_dist_list.append(embedding_dist)

  min_dist = min(embedding_dist_list)
  idx_min = embedding_dist_list.index(min_dist)
  print(min_dist)
  print(idx_min)

  return name_list[idx_min]


# Thread for QtSignal
class Thread(QThread):
  changePixmap = qtsignal(QImage)

  def run(self):
    self.isRunning = True
    cap = cv.VideoCapture(0)

    while self.isRunning:
      ret, frame = cap.read()

      if ret:
        rgbImage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  
        image = Image.fromarray(rgbImage)
        boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
        face, probs = mtcnn(image, return_prob=True)
        try:
          boxes = list(boxes)
          if boxes != []:
            for idx, box in enumerate(boxes):
              xmin, ymin, xmax, ymax = tuple(box)
              xmin = int(xmin)
              ymin = int(ymin)
              xmax = int(xmax)
              ymax = int(ymax)
              cv.rectangle(rgbImage, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
              # face_detected = torch.tensor(rgbImage[ymin:ymax, xmin:xmax])
              # face_detected = cv.resize(face_detected, (240, 240))

              try:
                name = recognize_face(face)
                cv.putText(rgbImage, name, (xmin, ymin-5), 1, 2, (0, 255, 0), 2)
              except:
                pass
        except TypeError:
          pass
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 500, Qt.KeepAspectRatio)
        self.changePixmap.emit(p)

  def stop(self):
    self.isRunning = False
    self.quit()
    self.terminate()


class VideoContainer(QWidget):
  def __init__(self):
    super().__init__()
    self.title = 'Face Recognition'
    self.left = 100
    self.top = 100
    self.fwidth = 640
    self.fheight = 550
    self.th = None
    self.initUI()

  @qtslot(QImage)
  def setImage(self, image):
    self.label.setPixmap(QPixmap.fromImage(image))

  def startThread(self):
    self.th = Thread(self)
    self.th.changePixmap.connect(self.setImage)
    self.th.start()

  def stopThread(self):
    self.th.changePixmap.disconnect(self.setImage)
    self.th.stop()


  def initUI(self):
    self.setWindowTitle(self.title)
    self.setGeometry(self.left, self.top, self.fwidth, self.fheight)
    self.setFixedSize(640, 550)
    self.setWindowIcon(QIcon('icons/app_icon.png'))

    self.label = QLabel(self)
    self.label.resize(640, 520)

    # create a button
    self.startIcon = QIcon('icons/start_recording.png')
    self.startbutton = QPushButton(self.startIcon, 'Start Monitoring', self)
    self.startbutton.setGeometry(QRect(200, 250, 120, 40)) # x, y, width, height
    # self.startbutton.setStyleSheet('background-color: blue')
    self.startbutton.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
    self.startbutton.clicked.connect(self.startThread)
    self.startbutton.move(200, 500)

    # create a button
    self.stopIcon = QIcon('icons/stop_recording.png')
    self.stopbutton = QPushButton(self.stopIcon, 'Stop Monitoring', self)
    self.stopbutton.setGeometry(QRect(200, 250, 120, 40)) # x, y, widht, height
    self.stopbutton.setStyleSheet('background-color: red; color: white; cursor: pointer;')
    self.stopbutton.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
    self.stopbutton.clicked.connect(self.stopThread)
    self.stopbutton.move(360, 500)

    self.show()

if __name__ == '__main__':

  app = QApplication(sys.argv)
  ex = VideoContainer()
  ex.show()
  sys.exit(app.exec_())

