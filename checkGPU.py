import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import OrderedDict
import myNetwork as mnw
import time

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(train_set
    ,batch_size=100
    ,shuffle=True
)




torch.set_grad_enabled(True)
number_run = 1
holdall = np.zeros(number_run)
params = OrderedDict(lr=[0.01, 0.001], batch_size=[1000, 2000])

if torch.backends.mps.is_available():
    device ="mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = 'cpu'
print(device)


#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#print(device)

network = mnw.Network().to(device)
optimizer = optim.Adam(network.parameters(), lr=0.01)
batch = next(iter(train_loader))
startTime = time.time()
for epoch in range(number_run):
    images, labels = batch
    images = images.to(device=device, dtype=torch.float32)
    if device == 'cuda':
        labels = labels.type(torch.LongTensor)
    labels = labels.to(device=device)  # , dtype=torch.float32)
    preds = network(images)
    loss = F.cross_entropy(preds, labels)
    holdall[epoch] = get_num_correct(preds, labels)
    optimizer.zero_grad()
    loss.backward() # Calculating the gradients
    optimizer.step()
#    if epoch % 10 == 0:
#        print(epoch)
print(time.time()-startTime)
plt.plot(holdall)
plt.show()