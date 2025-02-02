import time
import itertools
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import optim

from datasets import load_FashionMNIST
from helpers import SoftPlus, SoftHingeLoss, compute_accuracy_objective
from models import FCNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


mps_device = torch.device("mps")

trainloader, testloader = load_FashionMNIST("/Users/raphaelbordas/Code/sandbox_deep_learning/data")

n_layers = 3
# hh = np.array([10, 14, 19, 27, 37, 52, 72, 100, 139, 193, 268, 373]) #, 518, 720, 1000, 1390])
hh = np.array([10, 19, 37, 72, 139, 193, 373, 518, 720, 1000, 1390])
# hh = np.array([100, 300], dtype=int)
# alphas = np.array([1, ])
# alphas = np.array(
#     [1e-4, 3.2e-4, 1e-3, 3.2e-3, 1e-2, 3.2e-2, 1e-1, 3.2e-1, 1e0, 3.2e0, 1e1, 3.2e1, 1e2, 3.2e2, 1e3, 3.2e3, 1e4, 3.2e4,
#      1e5, 3.2e5, 1e6, 3.2e6, 1e7, ]
# )
alphas = 1 / np.sqrt(hh)

runs = np.arange(10) + 1
# hyperparams_runs = np.array(list(itertools.product(hh, alphas, runs)))
hyperparams_runs = np.stack([hh, alphas], axis=1)
results = pd.DataFrame(dict(
    h=np.repeat(hyperparams_runs[:, 0], 10),
    alpha=np.repeat(hyperparams_runs[:, 1], 10),
    runs=np.repeat(runs, hyperparams_runs.shape[0]),
    L=np.zeros(hyperparams_runs.shape[0] * 10),
    train_loss=np.zeros(hyperparams_runs.shape[0] * 10),
    train_acc=np.zeros(hyperparams_runs.shape[0] * 10),
    test_acc=np.zeros(hyperparams_runs.shape[0] * 10),
    n_epochs=np.zeros(hyperparams_runs.shape[0] * 10),
    param_sample_ratio=np.zeros(hyperparams_runs.shape[0] * 10),
    training_time=np.zeros(hyperparams_runs.shape[0] * 10),
))
# using format string to fill later with the parameters of the run
run_name = "fcnet_L{n_layers}_h{h}_alpha{alpha}_run{run}"

for idx, row in results.iterrows():
    h_, alpha_, run_ = row["h"], row["alpha"], row["runs"]
    start_time = time.time()
    net = FCNet(28 * 28, h=int(h_), L=n_layers, activation=SoftPlus(), copy_initial_params=False)
    net.to(mps_device)

    optimizer = optim.SGD(net.parameters(), lr=1)
    criterion = SoftHingeLoss(alpha=alpha_)
    f_omega0 = copy.deepcopy(net)
    f_omega0.eval()

    variances = []

    net.train()
    training_losses = []

    n_epochs = 100
    has_converged = True
    for epoch in tqdm(range(n_epochs),
                      desc=f"Training {idx + 1} / {len(hyperparams_runs)} [h = {h_}, alpha = {alpha_}]",
                      total=n_epochs, leave=True,
                      position=0):
        epoch_loss = 0.0
        num_batches = 0
        epoch_variance = []

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(mps_device), labels.to(mps_device)

            optimizer.zero_grad()
            with torch.no_grad():
                initial_output = f_omega0(inputs).squeeze()

            outputs = net(inputs).squeeze()
            loss = criterion(outputs - initial_output, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

            # checking is any of the samples does not meet the convergence criteria
            # in that case, we continue training
            if (alpha_ * (outputs - initial_output) * labels <= 1).any():
                has_converged = False

        avg_epoch_loss = epoch_loss / num_batches
        training_losses.append(avg_epoch_loss)

        # early stopping if criteria is met for all batches
        if has_converged and n_epochs > 1:
            break

    # save model
    torch.save(net.state_dict(),
               f"experiments/fig_4a/{run_name.format(n_layers=n_layers, h=h_, alpha=alpha_, run=run_)}.pth")
    torch.save(f_omega0.state_dict(),
               f"experiments/fig_4a/{run_name.format(n_layers=n_layers, h=h_, alpha=alpha_, run=run_)}_omega0.pth")

    # store results
    n_params = count_parameters(net)
    n_samples = len(trainloader.dataset)
    results.loc[idx, "h"] = h_
    results.loc[idx, "alpha"] = alpha_
    results.loc[idx, "L"] = n_layers
    results.loc[idx, "run"] = run_
    results.loc[idx, "train_loss"] = avg_epoch_loss  # average loss over the last epoch
    results.loc[idx, "n_epochs"] = len(training_losses)
    results.loc[idx, "train_acc"] = compute_accuracy_objective(trainloader, net, f_omega0, mps_device)
    results.loc[idx, "test_acc"] = compute_accuracy_objective(testloader, net, f_omega0, mps_device)
    results.loc[idx, "param_sample_ratio"] = n_params / n_samples
    results.loc[idx, "training_time"] = time.time() - start_time
    results.to_csv(f"experiments/fig_4a/results_fcnet_L{n_layers}.csv", index=False)
