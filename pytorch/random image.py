import numpy as np
import cv2

def bitOperation(hpos, vpos):
    img1 = cv2.imread('test/piano3.png')
    img2 = cv2.imread('test/mark.png')

    rows, cols, channels = img2.shape
    roi = img1[vpos:rows+vpos, hpos:cols+hpos]

    img2gray= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 266, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)
    img1[vpos:rows+vpos, hpos:cols+hpos] = dst

    cv2.imshow('result', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


mark_ls_w = [13,30,60,90,108,155,172,202,225,250,280,298]
mark_ls_h = [250,150,250,150,250,250,150,250,150,250,150,250]

octave = 332

for i in range(3):
    for j in range(len(mark_ls_w)):
        mark_w = octave * i + mark_ls_w[j]
        mark_h = mark_ls_h[j]

        # mark 표시 좌표 확인
        # print(mark_w, end=" ")
        # print(mark_h)
        # bitOperation(mark_w, mark_h)