# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 21:29:14 2016

@author: luca_
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread(r'img\text.jpg', cv2.IMREAD_GRAYSCALE)
        
plt.imshow(im, cmap='Greys_r')