import copy
import torch
from torch import optim

from datasets import load_FashionMNIST
from helpers import SoftPlus, SoftHingeLoss
from models import FCNet
from ntk_freeze import lazy_train, compute_accuracy

trainloader, testloader = load_FashionMNIST("/Users/raphaelbordas/Code/sandbox_deep_learning/data")

mps_device = torch.device("mps")

net = FCNet(28 * 28, h=100, L=3, activation=SoftPlus(), device=mps_device)
net.to(mps_device)

optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = SoftHingeLoss(alpha=1)
# f_omega0 = copy.deepcopy(net)
# f_omega0.eval()

lazy_train(
    net,
    trainloader,
    criterion,
    lr=0.01,
    device=mps_device
)

print(compute_accuracy(net.to("cpu"), testloader))
