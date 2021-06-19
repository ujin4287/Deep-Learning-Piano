import random
import cv2

import threading
import time

mark_ls_w = [24, 42, 71, 102, 119, 166, 184, 213, 237, 261, 292, 309]
mark_ls_h = [250, 150, 250, 150, 250, 250, 150, 250, 150, 250, 150, 250]

octave = 332
t=5

def show_img(left_hand, right_hand, finger_left, finger_right):

    left_list_temp=[]
    right_list_temp=[]

    for i in range(0, left_hand ): # 0~13 '''''''''''''''''''''''''
        left_list_temp.append(i)

    for i in range(0, right_hand ): # '''''''''''''''''''''''''''
        right_list_temp.append(i)

    left_out = random.sample(left_list_temp, finger_left)
    right_out = random.sample(right_list_temp, finger_right)

    left_out.sort()
    right_out.sort()

    # 왼손 시작
    temp1 = 36 - left_hand-right_hand # 0번째 배열부터 이다.
    left_start = random.randrange(0, temp1+1) #'''''''''''''''''''''''''''''''''

    # 오른손 시작
    temp2 = left_start + left_hand
    right_start = random.randrange(temp2, 36-right_hand+1) #'''''''''''''''''''''''''''''''

    #print("left : start - end")
    #print(left_start, left_start + left_hand - 1)
    #print("right : start - end:")
    #print(right_start, right_start + right_hand - 1)

    left_position = []
    right_position = []

    #print("left_position")
    for i in left_out:
        #print("left",i)
        #print(left_start + i)
        left_position.append(left_start + i)

    #print("right_position")
    for i in right_out:
        #print("right", i)
        #print(right_start + i)
        right_position.append(right_start + i)

    print(left_position,right_position)

    ##### 이미지 출력 #####

    #img1 = cv2.imread('test/piano3.png')

    # 왼손
    # 옥타브
    #print("left")
    for k in left_position:
        #print(k)
        mark_w = octave * (k//12) + mark_ls_w[k%12]
        mark_h = mark_ls_h[k%12]
        #cv2.circle(img1, (mark_w, mark_h), 10, (0, 0, 255), -1)

    # 오른손
    #print("right")
    for k in right_position:
        #print(k)
        mark_w = octave * (k//12) + mark_ls_w[k%12]
        mark_h = mark_ls_h[k%12]
        #cv2.circle(img1, (mark_w, mark_h), 10, (255, 0, 0), -1)

    #cv2.imshow('img', img1)
    #cv2.waitKey(t*1000)
    #cv2.destroyAllWindows()



##### 반복 #####

count = 0

while True:

    left_hand = random.randrange(0, 14) #'''''''''''''''''''''''' 안치는 것 부터 1개에서 13개
    right_hand = random.randrange(0, 14)

    finger_left = 0
    finger_right = 0

    if left_hand > 4:
        finger_left = random.randrange(0, 5)
    elif left_hand <= 4:
        finger_left = random.randrange(0, left_hand+1) #'''''''''''''''''''''''''

    if right_hand > 4:
        finger_right = random.randrange(0, 5)
    elif right_hand <= 4:
        finger_right = random.randrange(0, right_hand+1) #''''''''''''''''''''''''''''''


    if (finger_left == 0) and (finger_right == 0):
        #print("empty")
        continue

    else:
        count += 1
        #print("count = ", count)
        show_img(left_hand, right_hand, finger_left, finger_right)

    if count >= 1000:
        break