import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from resnet2 import resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
PATH = './save_model/piano_model_210504.pth'
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

if __name__ == '__main__':
    img_path = '0_a_27.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    detect(original_image)
    plt.imshow(original_image)