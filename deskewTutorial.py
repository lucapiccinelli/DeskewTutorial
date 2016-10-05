import cv2
import matplotlib.pyplot as plt

#caricamento dell'immagine da disco
im = cv2.imread(r'img\text.jpg', cv2.IMREAD_GRAYSCALE)

h, w = im.shape
#ridimensionamento dell'immagine
im = cv2.resize(im, (w/4,h/4))

#canny edge detector
im = cv2.Canny(im, 100, 200)

plt.imshow(im, cmap='Greys_r')