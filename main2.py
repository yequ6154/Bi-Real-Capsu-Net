import torch
import random
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.nn.functional as func
from torch.autograd import Variable
# import torch.autograd as grad
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import math
from birealcapsnet2 import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch bi-real capsulenet' )
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--lamda', type=float, default=0.5, help='learning rate')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--m_plus', type=float, default=0.9)
parser.add_argument('--m_minus', type=float, default=0.1)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()
mse_loss = nn.MSELoss(reduction='none')

def margin_loss(x, labels, lamda=0.5, m_plus=0.9, m_minus=0.1):
        v_c = torch.norm(x, dim=2, keepdim=True)
        tmp1 = func.relu(m_plus - v_c).view(x.shape[0], -1) ** 2
        tmp2 = func.relu(v_c - m_minus).view(x.shape[0], -1) ** 2
        loss_ = labels*tmp1 + lamda*(1-labels)*tmp2
        loss_ = loss_.sum(dim=1)
        return loss_
    
def reconst_loss(recnstrcted, data):
        loss = mse_loss(recnstrcted.view(recnstrcted.shape[0], -1), data.view(recnstrcted.shape[0], -1))
        return 0.4 * loss.sum(dim=1)
    
def loss(x, recnstrcted, data, labels, lamda=0.5, m_plus=0.9, m_minus=0.1):
        loss_ = margin_loss(x, labels, lamda, m_plus, m_minus) + reconst_loss(recnstrcted, data)
        return loss_.mean()


# In[15]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
# lr = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# # torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
# lambda1 = lambda: epoch: lr * 0.5**(epoch // 10)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# In[16]:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data augmentation
crop_scale = 0.08
lighting_param = 0.1
train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data2/',train=True,download=True,transform=train_transforms),batch_size=args.batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data2/',train=False,download=True,transform=test_transforms),batch_size=args.batch_size,shuffle=True)


def accuracy(indices, labels):
    correct = 0.0
    for i in range(indices.shape[0]):
        if float(indices[i]) == labels[i]:
            correct += 1
    return correct

def test(model, test_loader, loss, lamda=0.5, m_plus=0.9, m_minus=0.1):
  test_loss = 0.0
  correct = 0.0
  with torch.no_grad():
    for batch_idx, (data, label) in enumerate(test_loader):
      data, labels = data.to(device), one_hot(label.to(device))
      outputs, masked_output, recnstrcted, indices = model(data)
  #     if batch_idx == 9:
  #       print("test: ", indices)
      loss_test = model.loss(outputs, recnstrcted, data, labels, lamda, m_plus, m_minus)
      test_loss += loss_test.data
      indices_cpu, labels_cpu = indices.cpu(), label.cpu()
  #     for i in range(indices_cpu.shape[0]):
  #         if float(indices_cpu[i]) == labels_cpu[i]:
  #             correct += 1
      correct += accuracy(indices_cpu, labels_cpu)
  #     if batch_idx % 100 == 0:
  #        print("batch: ", batch_idx, "accuracy: ", correct/len(test_loader.dataset))
  #         print(indices_cpu)
    print("\nTest Loss: ", test_loss/len(test_loader.dataset), "; Test Accuracy: " , correct/len(test_loader.dataset) * 100,'\n')


log_dir = '/content/drive/My Drive/cifar_model.pth'  # 模型保存路径
def train(train_loader, model):
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    lambda1 = lambda epoch: 0.5**(epoch // 10)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.96)
    
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')
               
    for epoch in range(start_epoch+1, args.epochs):
      for batch_idx, (data, label_) in enumerate(train_loader):
        data, label = data.to(device), label_.to(device)
        labels = one_hot(label)
        optimizer.zero_grad()
        outputs, masked, recnstrcted, indices = model(data, labels)
        loss_val = model.loss(outputs, recnstrcted, data, labels, args.lamda, args.m_plus, args.m_minus)
        loss_val.backward()
        optimizer.step()
        if batch_idx%10 == 0:
          outputs, masked, recnstrcted, indices = model(data)
          loss_val = model.loss(outputs, recnstrcted, data, labels, args.lamda, args.m_plus, args.m_minus)
          print("epoch: ", epoch, "batch_idx: ", batch_idx, "loss: ", loss_val, "accuracy: ", accuracy(indices, label_.cpu())/indices.shape[0])
      test(model, test_loader, loss, args.lamda, args.m_plus, args.m_minus)
      lr_scheduler.step()
      state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
      torch.save(state, log_dir)   


if __name__ == '__main__':
    train(train_loader, model)

# Hard-Training
#print("\n\n\n\nHard-Training\n")
#train(train_loader, model, num_epochs=100, lr=0.001, batch_size=256, lamda=0.8, m_plus=0.95,  m_minus=0.05)

