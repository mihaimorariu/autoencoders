from typing import Tuple

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential, Sigmoid
from torch.nn.functional import mse_loss
from torch.optim import Adam


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
