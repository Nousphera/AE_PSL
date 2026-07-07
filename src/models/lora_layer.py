import math

import torch
from torch import nn

# TODO: check for correctness
class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 4, alpha: int = 1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # Freeze original weights
        # self.weight = nn.Parameter(original_layer.weight.clone().detach(), requires_grad=False)
        # self.bias = None
        # if original_layer.bias is not None:
        #     self.bias = nn.Parameter(original_layer.bias.clone().detach(), requires_grad=False)
        # Optimized Code
        self.weight = original_layer.weight
        self.weight.requires_grad = False

        # If original_layer had a bias
        if original_layer.bias is not None:
            self.bias = original_layer.bias
            self.bias.requires_grad = False

        # Trainable LoRA params
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_a = nn.Parameter(torch.randn(rank, self.in_features))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        original_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_a.t()) @ self.lora_b.t()
        return original_out + (lora_out * self.scaling)