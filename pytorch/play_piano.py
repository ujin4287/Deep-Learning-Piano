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

def play_piano(predicted_value):

        i = predicted_value

        if i == 0:
            print("0")
            C1.play()
        if i == 1:
            print("1")
            Db1.play()
        if i == 2:
            print("2")
            D1.play()
        if i == 3:
            print("3")
            Eb1.play()
        if i == 4:
            print("4")
            E1.play()
        if i == 5:
            print("5")
            F1.play()
        if i == 6:
            print("6")
            Gb1.play()
        if i == 7:
            print("7")
            G1.play()
        if i == 8:
            print("8")
            Ab1.play()
        if i == 9:
            print("9")
            A1.play()
        if i == 10:
            print("10")
            Bb1.play()
        if i == 11:
            print("11")
            B1.play()

        if i == 12:
            print("12")
            C2.play()
        if i == 13:
            print("13")
            Db2.play()
        if i == 14:
            print("14")
            D2.play()
        if i == 15:
            print("15")
            Eb2.play()
        if i == 16:
            print("16")
            E2.play()
        if i == 17:
            print("17")
            F2.play()
        if i == 18:
            print("18")
            Gb2.play()
        if i == 19:
            print("19")
            G2.play()
        if i == 20:
            print("20")
            Ab2.play()
        if i == 21:
            print("21")
            A2.play()
        if i == 22:
            print("22")
            Bb2.play()
        if i == 23:
            print("23")
            B2.play()

        if i == 24:
            print("24")
            C3.play()
        if i == 25:
            print("25")
            Db3.play()
        if i == 26:
            print("26")
            D3.play()
        if i == 27:
            print("27")
            Eb3.play()
        if i == 28:
            print("28")
            E3.play()
        if i == 29:
            print("29")
            F3.play()
        if i == 30:
            print("30")
            Gb3.play()
        if i == 31:
            print("31")
            G3.play()
        if i == 32:
            print("32")
            Ab3.play()
        if i == 33:
            print("33")
            A3.play()
        if i == 34:
            print("34")
            Bb3.play()
        if i == 35:
            print("35")
            B3.play()
        if i == 36:
            print("None")
            sleep(set_time)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from resnet2 import resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
PATH = './save_model/piano_model_210520.pth'
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

set_time = 0.5
capture = cv2.VideoCapture(1)
capture.set(3, 640) #가로길이
capture.set(4, 480) #세로길이



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

    while True:
        start_time = time.time()
        video_capture()
        # img_path = '0_a_27.jpg'
        img_path = 'temp.jpg'
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        predicted = detect(original_image)
        # print(predicted)
        predicted_value = torch.argmax(predicted)
        if predicted_value != 0:
            print(predicted_value)
            play_piano(predicted_value)
        else:
            print("None")

