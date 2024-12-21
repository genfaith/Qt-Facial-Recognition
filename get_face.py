import cv2 as cv
import sys
import numpy as np

from PySide6.QtWidgets import QWidget, QLabel, QApplication, QPushButton
from PySide6.QtCore import QThread, Qt, Signal as qtsignal, Slot as qtslot, QRect
from PySide6.QtGui import QImage, QPixmap, QCursor, QIcon
import PySide6.QtCore as QtCore

from facenet_pytorch import MTCNN
from PIL import Image

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(select_largest=False, post_process=False)

INCEPTION_HEIGHT, INCEPTION_WIDTH = 224, 224

# def fit_img_to_inception_input(captured_face_arr):
#   img_height, img_width = captured_face_arr.shape[:2]


#   top_padding = abs((INCEPTION_HEIGHT - img_height) // 2)
#   bottom_padding = abs(INCEPTION_HEIGHT - img_height - top_padding)
#   left_padding = abs((INCEPTION_WIDTH - img_width) // 2)
#   right_padding = abs(INCEPTION_WIDTH - img_width - left_padding)

#   # Any of the values above may be negative

#   # Creating a zero-padded arrau of target size (filled with black)
#   padded_image = np.zeros((INCEPTION_HEIGHT, INCEPTION_WIDTH, 3), dtype=np.uint8)
#   print(top_padding, img_height)
#   padded_image[top_padding:top_padding + img_height, left_padding:left_padding+img_width, :] = captured_face_arr

#   return padded_image


# Thread for QtSignal
class Thread(QThread):
  changePixmap = qtsignal(QImage)

  def run(self):
    self.isRunning = True
    cap = cv.VideoCapture(0)

    pic_idx = 0

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

              # padded_captured_face = fit_img_to_inception_input(frame[ymin:ymax, xmin:xmax, :])

              resized_img = cv.resize(frame[ymin:ymax, xmin:xmax], dsize=(INCEPTION_HEIGHT, INCEPTION_WIDTH))
              cv.imwrite(f'data/captures/faromika_ifeoluwa/{pic_idx:02d}.jpg', resized_img)
              print(resized_img.shape)

              pic_idx+=1
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

