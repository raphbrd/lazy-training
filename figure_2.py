import torch

from datasets import load_FashionMNIST
from models import FCNet
from helpers import SoftPlus

trainloader, testloader = load_FashionMNIST("/Users/raphaelbordas/Code/sandbox_deep_learning/data")

images, labels = next(iter(testloader))

results = {
    "h_268": {
        "run_numbers": list(range(11)),
        "h": 268,
        "alpha": 0.06108472217815261,
    },
    "h_373": {
        "run_numbers": list(range(11)),
        "h": 373,
        "alpha": 0.05177803730784977,
    },
    "h_518": {
        "run_numbers": list(range(11)),
        "h": 518,
        "alpha": 0.04393747751637468,
    },
    "h_720": {
        "run_numbers": list(range(11)),
        "h": 720,
        "alpha": 0.037267799624996496,
    },
    "h_1000": {
        "run_numbers": list(range(11)),
        "h": 1000,
        "alpha": 0.03162277660168379,
    },
}

variances = dict()
for key, value in results.items():
    print(key)
    outs = []
    for idx in value["run_numbers"]:
        out_run = []
        for images, labels in testloader:
            net = FCNet(28 * 28, h=value["h"], L=1, activation=SoftPlus(), copy_initial_params=False)
            net.load_state_dict(
                torch.load(f"experiments/fig_4a/fcnet_L1_h{value['h']}.0_alpha{value['alpha']}_run{idx}.0.pth"))

            net0 = FCNet(28 * 28, h=value["h"], L=1, activation=SoftPlus(), copy_initial_params=False)
            net0.load_state_dict(
                torch.load(f"experiments/fig_4a/fcnet_L1_h{value['h']}.0_alpha{value['alpha']}_run{idx}.0_omega0.pth"))

            with torch.no_grad():
                out = net(images).squeeze()
                out0 = net0(images).squeeze()
                out = value["alpha"] * (out - out0)
                if out.shape[0] == 128:
                    out_run.append(out)
        outs.append(out_run)
    outs_shaped = torch.cat([torch.cat(o) for o in outs])
    variance = outs_shaped.sub(outs_shaped.mean(0)).pow(2).mean()
    print(variance)  # tensor(1.4590e-05)
    variances[key] = variance
