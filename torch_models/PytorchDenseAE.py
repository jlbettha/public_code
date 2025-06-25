"""
Created on Sun Dec  5 12:31:08 2021

@author: jlb235
"""

import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt


# PyTorch Auto-Encoder class
class PytorchDenseAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def main() -> None:
    # Transform images to tensor
    tensor_transform = transforms.ToTensor()

    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=tensor_transform
    )
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    model = PytorchDenseAE()
    loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20
    outputs = []
    losses = []
    for epoch in range(epochs):
        for img, _ in loader:

            img = img.reshape(-1, 28 * 28)

            model_out = model(img)

            loss = loss_function(model_out, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss)

        outputs.append((epochs, img, model_out))

    plt.plot(losses)
    plt.xlabel("iters")
    plt.ylabel("loss")

    for i, _, img, model_out in enumerate(outputs):
        img = img.reshape(-1, 28, 28)
        plt.imshow(img)

        model_out = model_out.reshape(-1, 28, 28)
        plt.imshow(img)


if __name__ == "__main__":
    main()
