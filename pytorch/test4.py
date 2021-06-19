import torch
from torchvision import transforms

a = torch.zeros(1, 1)
b = torch.zeros(1, 1)

num = torch.cat((a, b), dim=1)

print(num)
print(num.shape)
print("")

num = num.numpy()
print(num)
print(num.shape)