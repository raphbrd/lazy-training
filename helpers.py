import torch
import torch.nn as nn


class SoftPlus(nn.Module):
    def __init__(self, beta=5, a=1.404):
        """ Softplus activation function. A multiplication factor of a = 1.404 is applied.

        This factor is used to normalize the preactivation and is computed as follows:
        ```{python}
        a = torch.nn.functional.softplus(torch.randn(100000, dtype=torch.float64), beta).pow(2).mean().rsqrt().item()
        ```
        WARNING: the factor is currently hard-coded in this function to avoid recomputing it at each call.

        Parameters
        ----------
        beta: float
            smooth factor, default is 5 as recommended in the paper
        a: float
            multiplication factor, default is 1.404 as recommended in the paper
        """
        super().__init__()
        self.a = a
        self.softplus = nn.Softplus(beta=beta)

    def forward(self, x):
        return self.softplus(x) * self.a


class SoftHingeLoss(nn.Module):
    def __init__(self, alpha, beta=20):
        """ Soft-hinge loss function

        Parameters
        ----------
        alpha: main factor, see the paper
        beta: smooth factor, default is 20 as recommended in the paper
        """
        super().__init__()
        self.softplus = nn.Softplus(beta=beta)
        self.alpha = alpha

    def forward(self, f, labels):
        return torch.mean(self.softplus(1 - self.alpha * f * labels)) / self.alpha


def compute_accuracy_objective(dataloader, f_trained, f_initial, model_device):
    """ Compute the accuracy when training with the objective function F(omega, x) = f(omega, x) - f(omega0, x) """
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        with torch.no_grad():
            initial_output = f_initial(inputs.to(model_device)).squeeze().to("cpu")
            outputs = f_trained(inputs.to(model_device)).squeeze().to("cpu")
        logits = outputs - initial_output
        predictions = torch.sign(logits)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100 * correct / total
