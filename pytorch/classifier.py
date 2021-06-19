import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10        # pytorch는 CIFAR10 갖고있음
from torch.utils.data import DataLoader

from resnet2 import test
import argparse
from torch.utils.tensorboard import SummaryWriter


### 무슨역할을 하는 부분인가요??
parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--batch_size_test', default=100, help='')
parser.add_argument('--num_worker', default=4, help='')
parser.add_argument('--logdir', type=str, default='logs', help='')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')

### Training dataset은 4로 padding한 이후에 32의 크기로 random cropping을 하고 horizontal flip을 랜덤하게 수행한다.
### 상하좌우로 padding을 해줘서 32의 크기가 된다는 건가요?
# @ padding 이후 random crop을 32*32로 한다고 되어 있네요

transforms_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(), # 0.5 기본
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



# Train이 True이면 training dataset을 불러오는 것이며 50000개의 이미지 데이터를 가져온다.
# Train이 False이면 testing dataset을 불러오는 것이고 10000개의 이미지 데이터를 가져온다.

dataset_train = CIFAR10(root='../data', train=True,
                        download=True, transform=transforms_train)
dataset_test = CIFAR10(root='../data', train=False,
                       download=True, transform=transforms_test)

# DataLoader는 mini batch 사이즈만큼 호출될 때마다 이미지와 라벨을 가져오는 함수이다.
# 어떤 데이터셋에서 mini batch를 가져올지 인자로 넣어주면 된다.
train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_worker)
test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test,
                         shuffle=False, num_workers=args.num_worker)




# there are 10 classes so the dataset name is cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Making model..')


# 모델에 포함되어있는 parameter의 수 출력 464154
net = test()
net = net.to(device)
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('The number of parameters of model is', num_params)
# print(net)



### ??
# @ resume 뜻이 재개한다는 뜻 → 이전에 학습시킨 모델을 불러와서 추가 학습
if args.resume is not None:
    checkpoint = torch.load('./save_model/' + args.resume)
    net.load_state_dict(checkpoint['net'])



# SGD의 learning rate는 처음에 0.1로 설정하고
# 32000번 업데이트를 하고나면 0.01로
# 48000번 업데이트를 하고나면 0.001로 설정한다.
### 근데 여기서 총 몇개인줄 알고 이걸 설정하나요? 보통 설정하는 비율이 있나요?
### 64000 step? 어떻게 나온건가요? 그냥 정해준건가요?
# 하이퍼 파라미터는 모델링할 때 사용자가 직접 세팅해주는 값
# 하이퍼 파라미터는 정해진 최적의 값이 없습니다.

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=1e-4)  # weight_decay 가중치 감소, 오버피팅을 막기 위해 특정값을 손실함수에 더해주는 것

decay_epoch = [32000, 48000]
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=decay_epoch, gamma=0.1) # lr에 0.1곱
# The SummaryWriter class is your main entry to log data for consumption and visualization by TensorBoard.
writer = SummaryWriter(args.logdir)




# net을 train()을 통해 학습 모드로 전환해준다.
def train(epoch, global_steps):
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    # train_loader에서 mini batch씩 데이터를 꺼내와서 loss를 계산
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        global_steps += 1
        step_lr_scheduler.step()    ### 매 mini batch마다 한 번씩 수행?
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # 역전파 단계 전에, Optimizer 객체를 사용하여 (모델의 학습 가능한 가중치인)
        # 갱신할 변수들에 대한 모든 변화도를 0으로 만듭니다. 이렇게 하는 이유는
        # 기본적으로 .backward()를 호출할 때마다 변화도가 버퍼(buffer)에 (덮어쓰지 않고)
        # 누적되기 때문입니다.
        optimizer.zero_grad()   # 초기화
        loss.backward()         # 역전파 단계: 모델의 매개변수에 대한 손실의 변화도를 계산합니다
        optimizer.step()        # Optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100 * correct / total
    print('train epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
        epoch, batch_idx, len(train_loader), train_loss / (batch_idx + 1), acc))

    writer.add_scalar('log/train error', 100 - acc, global_steps)
    return global_steps


def test(epoch, best_acc, global_steps):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100 * correct / total
    print('test epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
        epoch, batch_idx, len(test_loader), test_loss / (batch_idx + 1), acc))

    writer.add_scalar('log/test error', 100 - acc, global_steps)

    if acc > best_acc:
        print('==> Saving model..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('save_model'):
            os.mkdir('save_model')
        torch.save(state, './save_model/ckpt.pth')
        best_acc = acc

    return best_acc


if __name__ == '__main__':
    best_acc = 0
    epoch = 0
    global_steps = 0

    if args.resume is not None:
        test(epoch=0, best_acc=0)
    else:
        while True:
            epoch += 1
            global_steps = train(epoch, global_steps)
            best_acc = test(epoch, best_acc, global_steps)
            print('best test accuracy is ', best_acc)

            if global_steps >= 64000:
                break

### 총 몇 epoch 도는지 어떻게 아나요?
# 391* 164 = 64124