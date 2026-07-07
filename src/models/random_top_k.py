import torch.nn as nn
import numpy as np
import torch


# From https://github.com/Federico-Alvetreti/Split-Learning/blob/main/methods/random_top_k.py

# Selects the Top-K of a tensor with a random portion
class RandomTopKModifier(nn.Module):
    def __init__(self, rate: float, random_portion: float = 0.1):
        super(RandomTopKModifier, self).__init__()
        self.rate = rate
        self.random_portion = random_portion

    def select_top_k(self, x: torch.Tensor):

        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        k = max(1, round(self.rate * x_flat.shape[1]))
        sample_dim = x_flat.shape[1]
        _, top_k_indices = torch.topk(torch.abs(x_flat), k, sorted=False)

        probs = torch.full_like(x_flat.real, self.random_portion / (sample_dim - k))
        probs = torch.scatter(probs, 1, top_k_indices, (1 - self.random_portion) / k)

        selected_indices = torch.multinomial(probs, k)
        mask = torch.scatter(torch.zeros_like(x_flat), 1, selected_indices, 1)

        return mask.view(*x.shape) * x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.select_top_k(x)


    # Function to get the compression level of this method
    def get_compression_level(self, batch_elements):


        # lower rate means more compression
        # Compute forward and backward compression   (following https://arxiv.org/pdf/2305.18469)
        if self.rate == 1:
            forward_compression = 1
        else:
            forward_compression = self.rate * (1 + np.ceil(np.log2(batch_elements)) / 32)

        backward_compression = self.rate

        # Return average compression
        compression = (forward_compression + backward_compression) / 2
        compression_ratio = 1.0 / compression

        # print(f"RandomTopKModifier: forward_compression={forward_compression:.4f}, backward_compression={backward_compression:.4f}, average_compression={compression:.4f}, compression_ratio={compression_ratio:.4f}")
        return compression_ratio

