import torch
from torch import nn


"""Custom loss wrappers used during training.

The `CustomLoss` class exposes a small set of common regression losses
and supports masking missing target values by ignoring NaNs.
"""


class CustomLoss(nn.Module):
    """Wrapper around several loss functions with NaN-masking.

    Parameters
    ----------
    loss_function : str
        One of `MSE`, `RMSE`, `MAE`, `NSE_complement` selecting the loss to use.
    skip : int, optional
        Not used currently; kept for API compatibility.
    """

    def __init__(self, loss_function: str, skip: int = 0):
        super(CustomLoss, self).__init__()

        # Dict of all available loss functions
        loss_functions = {
            "MSE": self.MSE,
            "RMSE": self.RMSE,
            "MAE": self.MAE,
            "NSE_complement": self.NSE_complement,
        }

        # Use this loss function
        self.loss_function = loss_functions[loss_function]

    def forward(self, y_true: torch.Tensor, y_predict: torch.Tensor) -> torch.Tensor:
        """Compute the masked loss between `y_true` and `y_predict`.

        NaN values in `y_true` are ignored via a boolean mask.
        """

        mask = ~torch.isnan(y_true)
        loss = self.loss_function(y_true, y_predict, mask)

        return loss

    # Mean square error
    def MSE(self, y_true: torch.Tensor, y_predict: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        # Mean square error
        loss = nn.MSELoss()
        mse = loss(y_true[mask], y_predict[mask])

        # Return output
        return mse

    # Mean absolute error
    def MAE(self, y_true: torch.Tensor, y_predict: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        # Mean absolute error
        loss = nn.L1Loss()
        mae = loss(y_true[mask], y_predict[mask])
        return mae

    # Root mean square error
    def RMSE(self, y_true: torch.Tensor, y_predict: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        # Root Mean Square Error
        rmse = self.MSE(y_true, y_predict, mask) ** 0.5

        return rmse

    # Complement to 1 of the Nash-Sutcliffe (or 1- Nash sutcliffe)
    def NSE_complement(self, y_true: torch.Tensor, y_predict: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        # Sum Square Error
        sse = torch.sum((y_true[mask] - y_predict[mask]) ** 2)

        # Sum of Square Difference around mean
        ssd = torch.sum((y_true[mask] - torch.mean(y_true[mask])) ** 2)

        # Minimize loss, so output should be sse/ssd, which is 1 - NSE
        return sse / ssd
