import torch

from helpers import SoftHingeLoss


def lazy_train(model, dataloader, criterion, epochs=10, lr=0.01, device='cpu'):
    # Store the NTK gradients
    ntk_grads = {}

    # Compute NTK once at initialization
    for name, param in model.named_parameters():
        ntk_grads[name] = torch.zeros_like(param).to(device)

    model.train()

    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            images = images.view(images.shape[0], -1)  # Flatten images
            labels = 2 * labels - 1  # Convert {0,1} to {-1,1}

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Compute gradients w.r.t initial parameters (Frozen NTK)
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    ntk_grads[name] += param.grad  # Accumulate NTK gradient

        # Perform a lazy update using the initial parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                # print(param.device)
                # print(model.initial_params[name].device)
                # print(ntk_grads[name].device)
                param.copy_(model.initial_params[name] - lr * ntk_grads[name])

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    return ntk_grads
