import argparse
import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer import Trainer
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential, Sigmoid
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor


class SimpleAutoencoder(LightningModule):
    def __init__(
        self,
        image_height: int = 28,
        image_width: int = 28,
        n_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.encoder = Sequential(
            Linear(image_width * image_height, n_hidden_dim),
            ReLU(),
        )
        self.decoder = Sequential(
            Linear(n_hidden_dim, image_width * image_height),
            Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        h = x.reshape(-1)

        h = self.encoder(h)
        y = self.decoder(h)

        y = y.reshape(x.shape)
        return y

    def training_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        batch: Tuple[Tensor, Tensor],
    ) -> Tensor:
        x, _ = batch
        x_pred = self.forward(x)
        loss = mse_loss(x_pred, x)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer


def main(args):
    logging.basicConfig()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = MNIST(
        root="../datasets",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # For speed purposes, train on a subset only.
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        sampler=RandomSampler(
            train_dataset,
            num_samples=3000,
        ),
    )

    # For speed purposes, test on a subset only.
    test_dataset = MNIST(
        root="../datasets",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=8,
        sampler=RandomSampler(
            test_dataset,
            num_samples=10,
        ),
    )

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    logging.info(f"Using device: {device}")

    if is_cuda:
        logging.info(f"CUDA detected: {torch.cuda.get_device_name(device)}")

    model = SimpleAutoencoder(n_hidden_dim=args.n_hidden_dim)
    model.train()

    trainer = Trainer(
        enable_checkpointing=False,
        max_epochs=args.max_epochs,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, test_loader)

    model.eval()
    plt.figure(figsize=(8, 4))
    n_samples = len(test_loader)

    for i, (x, _) in enumerate(test_loader):
        with torch.no_grad():
            x_pred = model(x)

        plt.subplot(2, n_samples, i + 1)
        plt.imshow(x.numpy().squeeze())
        plt.title("original")
        plt.gray()

        plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(x_pred.numpy().squeeze())
        plt.title("reconstructed")
        plt.gray()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=12345)
    parser.add_argument("--max-epochs", "-e", type=int, default=2)
    parser.add_argument("--n-hidden-dim", "-d", type=int, default=64)

    args = parser.parse_args()
    main(args)
