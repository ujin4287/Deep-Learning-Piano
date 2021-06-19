import torch
torch.cuda.get_device_name(0)
print(torch.cuda.is_available())
torch.__version__