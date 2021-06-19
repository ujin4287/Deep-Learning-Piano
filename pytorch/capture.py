import cv2
import time

capture = cv2.VideoCapture(1)

capture.set(3, 640) #가로길이
capture.set(4, 480) #세로길이

img_counter = 0
frame_set = []
start_time = time.time()

def video_capture(start_time,img_counter):

    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if time.time() - start_time >= 5:
            img_name = "./sample/opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_counter))
            start_time = time.time()
        img_counter += 1

video_capture(start_time,img_counter)

