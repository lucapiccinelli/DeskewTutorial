import cv2
import matplotlib.pyplot as plt

#importazione delle librerie necessarie
import numpy as np
import math

def houghSpace(im):
    maxTheta = 180
    #la larghezza dello spazio corrisponde alla massima angolatura presa in considerazione
    houghMatrixCols = maxTheta
    
    #dimensioni dell'immagine originale
    h, w = im.shape
    #non puo' esistere nell'immagine una distanza superiore alla diagonale
    rhoMax = math.sqrt(w * w + h * h) 
    #l'altezza dello spazio è il doppio della rho massima, per considerare anche 
    #le rho negative
    houghMatrixRows = int(rhoMax) * 2 + 1
    #le rho calcolate verranno traslate della metà dell'altezza per poter rappresentare nello spazio
    #anche le rho negative
    rhoOffset = houghMatrixRows/2
    
    #riscalature per passare da angoli a radianti
    degToRadScale = 0.01745329251994329576923690768489 # Pi / 180
    #$seno e coseno precalcolati
    rangemaxTheta = range(0,maxTheta)
    sin, cos = zip(*((math.sin(i * degToRadScale), math.cos(i * degToRadScale)) for i in rangemaxTheta))
    
    #inizializzazione dello spazio
    houghSpace = [0.0 for x in range(houghMatrixRows * houghMatrixCols)]
    
    #scorro tutta l'immagine originale
    for y in range(0, h):
        for x in range(0, w):
            #per ogni punto di bordo
            if im[y, x] > 0:
                #calcolo il suo fascio di rette...
                for theta in rangemaxTheta:
                    #... per ogni angolazione theta nello spazio, calcolo il relativo valore di rho
                    #... utilizzando la forma polare dell'equazione della retta
                    rho = int(round(x * cos[theta] + y * sin[theta] + rhoOffset))
                    
                    #una volta note le coordinate theta e rho, incremento il contatore dello spazio di Hough
                    # alla coordinata
                    c = rho * houghMatrixCols + theta
                    houghSpace[c] = houghSpace[c] + 1
    
    # normalizzazione tra 0 e 1
    m = np.max(houghSpace)
    houghSpace = houghSpace / m
    return np.reshape(houghSpace , (houghMatrixRows, houghMatrixCols)) #reshape in forma matriciale


#caricamento dell'immagine da disco
im = cv2.imread(r'img\text.jpg', cv2.IMREAD_GRAYSCALE)

h, w = im.shape
#ridimensionamento dell'immagine
im = cv2.resize(im, (w/4,h/4))

#canny edge detector
im = cv2.Canny(im, 100, 200)

#applicazione del calcolo dello spazio di hough all'immagine di input
hSpace = houghSpace(im)

#visualizzazione dello spazio risultante
h, w = hSpace.shape
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.imshow(hSpace, extent=[0,w,0,h])
plt.colorbar(cax, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.show()