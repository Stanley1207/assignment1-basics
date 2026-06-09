import torch
import torch.nn as nn

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class FFN(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None=None,
            dtype: torch.dtype | None=None,
    ):
        super().__init__()

        from cs336_basics.modules.linear import Linear

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))