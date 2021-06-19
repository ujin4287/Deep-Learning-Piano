import numpy as np
import cv2

img1 = cv2.imread('test/piano3.png')

mark_ls_w = [24,42,71,102,119,166,184,213,237,261,292,309]
mark_ls_h = [250,150,250,150,250,250,150,250,150,250,150,250]

octave = 332

for i in range(3):
    for j in range(12):
        mark_w = octave * i + mark_ls_w[j]
        mark_h = mark_ls_h[j]
        cv2.circle(img1, (mark_w, mark_h), 10, (0, 0, 255), -1)


cv2.imshow('img', img1)

cv2.waitKey()
cv2.destroyAllWindows()