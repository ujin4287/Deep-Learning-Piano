import glob
from PIL import Image

'''
image1 = Image.open('C:/Users/ujin4/PycharmProjects/untitled/sample/piano_1_1/*.jpg')
image1.show()

# 이미지의 크기 출력
print(image1.size)

# 이미지 자르기 crop함수 이용 ex. crop(left,up, right, down)
croppedImage = image1.crop((20, 70, 620, 160))

croppedImage.show()

print("잘려진 사진 크기 :", croppedImage.size)

croppedImage.save('C:/Users/ujin4/PycharmProjects/untitled/cropped_image/crop.jpg')
'''

###

images = glob.glob('C:/Users/ujin4/PycharmProjects/untitled/sample/piano_1_1/*.jpg')

### 파일 리스트 이름 가져오기
import re
import os

dir = 'C:/Users/ujin4/PycharmProjects/untitled/sample/piano_1_1'

temp = []

for pack in os.walk(dir):
    # print("pack:", pack[2])
    temp = pack[2]
    # print("pack:", pack[2][0])

# print(temp)
# print(file_name_list)

k = 0

for file in images:
    load_img = Image.open(file)
    # print(file) # C:/Users/ujin4/PycharmProjects/untitled/sample/piano_1_1\480_a_13_34.jpg

    croppedImage = load_img.crop((20, 70, 620, 155)) # 이미지 자르기 crop함수 이용 ex. crop(left,up, right, down) # 10, 10, 630, 160
    # croppedImage.show()

    croppedImage.save('C:/Users/ujin4/PycharmProjects/untitled/cropped_image/{}'.format(temp[k]))

    k += 1