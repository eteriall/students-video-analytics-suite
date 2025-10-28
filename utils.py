import cv2
from PyQt5.QtGui import QPixmap, QImage


def cv_to_qpixmap(img):
    if img is None:
        return QPixmap()
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)
