import random
import cv2
import time
from time import sleep

import pygame

pygame.init()
'''
WHITE = (255, 255, 255)
font = pygame.font.SysFont("comicsansms", 72)
text = font.render("Python Piano", True, (128, 128, 128))

pyscreen = pygame.display.set_mode((1024, 768), 0, 32)
pyscreen.fill(WHITE)
pyscreen.blit(text, (512 - text.get_width() // 2, 384 - text.get_height() // 2))
pygame.display.flip()
'''

C1 = pygame.mixer.Sound('./sound/Piano.mf.C1.mp3')
Db1 = pygame.mixer.Sound('./sound/Piano.mf.Db1.mp3')
D1 = pygame.mixer.Sound('./sound/Piano.mf.D1.mp3')
Eb1 = pygame.mixer.Sound('./sound/Piano.mf.Eb1.mp3')
E1 = pygame.mixer.Sound('./sound/Piano.mf.E1.mp3')
F1 = pygame.mixer.Sound('./sound/Piano.mf.F1.mp3')
Gb1 = pygame.mixer.Sound('./sound/Piano.mf.Gb1.mp3')
G1 = pygame.mixer.Sound('./sound/Piano.mf.G1.mp3')
Ab1 = pygame.mixer.Sound('./sound/Piano.mf.Ab1.mp3')
A1 = pygame.mixer.Sound('./sound/Piano.mf.A1.mp3')
Bb1 = pygame.mixer.Sound('./sound/Piano.mf.Bb1.mp3')
B1 = pygame.mixer.Sound('./sound/Piano.mf.B1.mp3')

C2 = pygame.mixer.Sound('./sound/Piano.mf.C2.mp3')
Db2 = pygame.mixer.Sound('./sound/Piano.mf.Db2.mp3')
D2 = pygame.mixer.Sound('./sound/Piano.mf.D2.mp3')
Eb2 = pygame.mixer.Sound('./sound/Piano.mf.Eb2.mp3')
E2 = pygame.mixer.Sound('./sound/Piano.mf.E2.mp3')
F2 = pygame.mixer.Sound('./sound/Piano.mf.F2.mp3')
Gb2 = pygame.mixer.Sound('./sound/Piano.mf.Gb2.mp3')
G2 = pygame.mixer.Sound('./sound/Piano.mf.G2.mp3')
Ab2 = pygame.mixer.Sound('./sound/Piano.mf.Ab2.mp3')
A2 = pygame.mixer.Sound('./sound/Piano.mf.A2.mp3')
Bb2 = pygame.mixer.Sound('./sound/Piano.mf.Bb2.mp3')
B2 = pygame.mixer.Sound('./sound/Piano.mf.B2.mp3')

C3 = pygame.mixer.Sound('./sound/Piano.mf.C3.mp3')
Db3 = pygame.mixer.Sound('./sound/Piano.mf.Db3.mp3')
D3 = pygame.mixer.Sound('./sound/Piano.mf.D3.mp3')
Eb3 = pygame.mixer.Sound('./sound/Piano.mf.Eb3.mp3')
E3 = pygame.mixer.Sound('./sound/Piano.mf.E3.mp3')
F3 = pygame.mixer.Sound('./sound/Piano.mf.F3.mp3')
Gb3 = pygame.mixer.Sound('./sound/Piano.mf.Gb3.mp3')
G3 = pygame.mixer.Sound('./sound/Piano.mf.G3.mp3')
Ab3 = pygame.mixer.Sound('./sound/Piano.mf.Ab3.mp3')
A3 = pygame.mixer.Sound('./sound/Piano.mf.A3.mp3')
Bb3 = pygame.mixer.Sound('./sound/Piano.mf.Bb3.mp3')
B3 = pygame.mixer.Sound('./sound/Piano.mf.B3.mp3')

global predicted_value
global previous_value

def play_piano():
    global predicted_value
    for i in predicted_value:

        if i == 0:
            print("C1")
            C1.play()
        if i == 1:
            print("Db1")
            Db1.play()
        if i == 2:
            print("D1")
            D1.play()
        if i == 3:
            print("Eb1")
            Eb1.play()
        if i == 4:
            print("E1")
            E1.play()
        if i == 5:
            print("F1")
            F1.play()
        if i == 6:
            print("Gb1")
            Gb1.play()
        if i == 7:
            print("G1")
            G1.play()
        if i == 8:
            print("Ab1")
            Ab1.play()
        if i == 9:
            print("A1")
            A1.play()
        if i == 10:
            print("Bb1")
            Bb1.play()
        if i == 11:
            print("B1")
            B1.play()

        if i == 12:
            print("C2")
            C2.play()
        if i == 13:
            print("Db2")
            Db2.play()
        if i == 14:
            print("D2")
            D2.play()
        if i == 15:
            print("Eb2")
            Eb2.play()
        if i == 16:
            print("E2")
            E2.play()
        if i == 17:
            print("F2")
            F2.play()
        if i == 18:
            print("Gb2")
            Gb2.play()
        if i == 19:
            print("G2")
            G2.play()
        if i == 20:
            print("Ab2")
            Ab2.play()
        if i == 21:
            print("A2")
            A2.play()
        if i == 22:
            print("Bb2")
            Bb2.play()
        if i == 23:
            print("B2")
            B2.play()

        if i == 24:
            print("C3")
            C3.play()
        if i == 25:
            print("Db3")
            Db3.play()
        if i == 26:
            print("D3")
            D3.play()
        if i == 27:
            print("Eb3")
            Eb3.play()
        if i == 28:
            print("E3")
            E3.play()
        if i == 29:
            print("F3")
            F3.play()
        if i == 30:
            print("Gb3")
            Gb3.play()
        if i == 31:
            print("G3")
            G3.play()
        if i == 32:
            print("Ab3")
            Ab3.play()
        if i == 33:
            print("A3")
            A3.play()
        if i == 34:
            print("Bb3")
            Bb3.play()
        if i == 35:
            print("B3")
            B3.play()
        if i == 36:
            print("None")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from resnet2 import resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
PATH = './save_model/piano_model_210605_50epoch.pth'
checkpoint = torch.load(PATH)
start_epoch = checkpoint['epoch'] + 1
# print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)

model = resnet()
model.load_state_dict(checkpoint['net'])
model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((360, 360))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

capture = cv2.VideoCapture(1)
capture.set(3, 640) #????????????
capture.set(4, 480) #????????????



def detect(original_image):

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # print(image)
    # print(image.shape)

    # Forward prop.
    predicted = model(image.unsqueeze(0))
    # print(predicted.shape)
    # print(predicted)
    predicted = (predicted > 0.5).float()

    # print(predicted)

    return predicted

set_time = 0.01

def video_capture():

    global start_time, num_imgs
    while True:
        # print("video")
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

        if time.time() - start_time >= set_time:
            img_name = "temp.jpg"
            cv2.imwrite(img_name, frame)
            break

if __name__ == '__main__':

    global previous_value
    global predicted_value

    previous_value = 36

    while True:
        start_time = time.time()
        video_capture()
        # img_path = '0_a_27.jpg'
        img_path = 'C:/Users/ujin4/PycharmProjects/untitled/temp.jpg'
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        predicted = detect(original_image)
        predicted = predicted.tolist()
        # print(predicted)
        predicted_value = [i for i, value in enumerate(predicted[0]) if value == 1]

        if predicted_value and (36 not in predicted_value) and len(predicted_value) <= 2 and previous_value != predicted_value:
            play_piano() # ????????? ?????? ??????
            previous_value = predicted_value
            print("predicted_value: ", predicted_value)
        else:
            previous_value = predicted_value
            # print("None : predicted_value: ", predicted_value)
        sleep(set_time)
