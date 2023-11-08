import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import OrderedDict
import myNetwork as mnw


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

#train_loader = torch.utils.data.DataLoader(train_set
#    ,batch_size=100
#    ,shuffle=True
#)




torch.set_grad_enabled(True)
number_run = 100
holdall = np.zeros(number_run)
params = OrderedDict(lr=[0.01, 0.001], batch_size=[100])
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for run in mnw.RunBuilder.get_runs(params):
    network = mnw.Network().to(device)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, shuffle=False)

    for epoch in range(number_run):
        for batchNumber, batch in enumerate(train_loader):
            images, labels = batch
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            optimizer.zero_grad()
            loss.backward() # Calculating the gradients
            optimizer.step()
        holdall[epoch] = 100 * (get_num_correct(preds, labels)/run.batch_size)
        if epoch % 10 == 0:
            print(epoch, run.lr, run.batch_size)
    plt.plot(holdall)
plt.show()