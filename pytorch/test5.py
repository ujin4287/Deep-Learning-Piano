import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('0_a_28_-1.jpg',0) # 흑백 이미지
edges = cv2.Canny(img, 100, 200) # edge 이미지

print(img)
print(img.shape) #'numpy.ndarray' object

print(edges)
print(edges.shape)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])


plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

from PIL import Image
import torchvision.transforms as transforms

def get_thick_edges(im_gray, steps_log):
    # Canny for edges detection
    thick_edges = np.zeros_like(im_gray)
    edges = cv2.Canny(im_gray, 100, 200)

    # Wiggle the image a bit to get thicker edges, closing holes
    for i in range(-4, 4):
        thick_edges += (np.roll(edges, i, 0) + np.roll(edges, i, 1))
    thick_edges = 1 - (thick_edges > 0).astype(np.uint8)

    # color_coverted = cv2.cvtColor(thick_edges, cv2.COLOR_BGR2RGB)
    pil_Image = Image.fromarray(thick_edges)

    # Log
    # steps_log.append(('Edges Detection + Thickening', expand2rgb(255 * thick_edges)))
    return pil_Image

thick_img = get_thick_edges(img, 1)

plt.plot(1), plt.imshow(thick_img,cmap = 'gray')
plt.title('Thick Image'), plt.xticks([]), plt.yticks([])
plt.show()