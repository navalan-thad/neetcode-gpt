import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.lin1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.lin2 = nn.Linear(512, 10)
        self.sig = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        z1 = self.lin1(images)
        a1 = self.relu(z1)
        self.dropout(a1)
        z2 = self.lin2(a1)
        a2 = self.sig(z2)

        return torch.round(a2, decimals=4)
