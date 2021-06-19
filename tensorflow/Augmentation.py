import numpy as np
import tensorflow as tf
import glob

# 랜덤시드 고정시키기
np.random.seed(5)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 데이터셋 불러오기
data_aug_gen = ImageDataGenerator(rescale=1. / 255,
                                  rotation_range=5,
                                  width_shift_range=0.05,
                                  height_shift_range=0.05,
                                  shear_range=0.1,####
                                  zoom_range=[1.0, 1.2],
                                  horizontal_flip=False,
                                  vertical_flip=False,
                                  fill_mode='nearest') # 큐빅/ 리니어


images = glob.glob('C:/Users/ujin4/PycharmProjects/untitled/sample/piano_1_total/*.jpg')
#print(images)


#####

import re
import os

dir = 'C:/Users/ujin4/PycharmProjects/untitled/sample/piano_1_total'

file_name_list = []

for pack in os.walk(dir):

    for i in pack[2]:
        numbers = re.findall("\d+", i)
        file_name_list.append(numbers[0])
        # print(numbers[0])

# print(file_name_list)

#####

with tf.device("gpu:0"):

    k = 0

    print("tf.keras code in this scope will run on GPU")
    for file in images:
        print(int(file_name_list[k]))

        if int(file_name_list[k]) < 2172 or (4111 < int(file_name_list[k]) and 5977 > int(file_name_list[k])): # 정답이 있는 이미지
            print("if", file)

            img = load_img(file, target_size=(360, 360))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            i = 0

            # 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야합니다.
            for batch in data_aug_gen.flow(x, batch_size=1,
                                           save_to_dir='C:/Users/ujin4/PycharmProjects/untitled/augmentation/sample/total/piano_1',
                                           save_prefix='%s' % (file_name_list[k]), save_format='jpg'):
                i += 1
                if i >= 10:
                    break
        else: # 정답이 없는 이미지
            print("el", file)

            img = load_img(file, target_size=(360, 360))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            i = 0

            # 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야합니다.
            for batch in data_aug_gen.flow(x, batch_size=1,
                                           save_to_dir='C:/Users/ujin4/PycharmProjects/untitled/augmentation/sample/total/piano_1',
                                           save_prefix='%s'%(file_name_list[k]), save_format='jpg'):
                i += 1
                if i >= 3:
                    break
        k += 1