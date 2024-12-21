# This script is for capturing image data

import cv2 as cv
import time

def capture():
  cap = cv.VideoCapture(0)

  file_count = 0

  rep_time = 0

  while True:
    ret, frame = cap.read()
    print(type(frame))

    if ret:
      cv.imshow('Capture', frame)
      if rep_time % 50 == 0:
        cv.imwrite(f'data/images/faromika_ifeoluwa/img_{file_count:03d}.jpg', frame)
        file_count += 1
    
    if file_count == 33:
      break
    
    if ord('q') == cv.waitKey(1):
      break
  
    rep_time += 1




























































    

  cv.destroyAllWindows()
  

if __name__ == '__main__':
  capture()