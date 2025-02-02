import torch
import torchvision

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """ Basic custom dataset class for PyTorch. """

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def load_FashionMNIST(root_path, batch_size=128, n_samples_training=10000):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    # data are assumed to be already downloaded in the root folder
    # otherwise, specify path and set download=True
    trainset = torchvision.datasets.FashionMNIST(root_path, train=True, download=False, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root_path, train=False, download=False, transform=transform)
    dataset = list(trainset) + list(testset)
    classes = sorted({y for x, y in dataset})

    torch.manual_seed(42)
    new_idx = torch.randperm(len(dataset))
    dataset = [dataset[i] for i in new_idx]
    x = torch.stack([x for x, y in dataset])
    y = torch.tensor([y for x, y in dataset])

    # transform y for binary classification
    y = (2 * (y % 2) - 1).type(x.dtype)
    # dict to track if the new class is a positive or negative class
    class_sign = {i: sign for i, sign in zip(classes, y.unique())}

    x_train = x[:n_samples_training]
    y_train = y[:n_samples_training]
    x_test = x[n_samples_training:]
    y_test = y[n_samples_training:]

    trainloader = torch.utils.data.DataLoader(
        CustomDataset(x_train, y_train),
        batch_size=batch_size,
    )
    testloader = torch.utils.data.DataLoader(
        CustomDataset(x_test, y_test),
        batch_size=batch_size,
    )

    return trainloader, testloader
