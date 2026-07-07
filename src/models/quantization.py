import copy

import torch.nn as nn
import torch
import torch.nn.functional as F

from utils.mpsl_utils import get_communication_size


# Simple Uniform Quantization
class _QuantizeFunction(torch.autograd.Function):

    @staticmethod
    def quantize(x: torch.tensor, n_bits: int):
        # Handle the case of complex tensor applying the quantization to both real and imaginary parts
        if torch.is_complex(x):
            real_q = _QuantizeFunction.quantize(x.real, n_bits)
            imag_q = _QuantizeFunction.quantize(x.imag, n_bits)
            return torch.complex(real_q, imag_q)

        # Get min and max
        x_min = x.min()
        x_max = x.max()

        # Get levels
        levels = 2 ** n_bits

        # Get the step
        scale = (x_max - x_min) / (levels - 1)

        # Quantize: map x to integers [0, levels-1]
        q_x = torch.round((x - x_min) / scale).clamp(0, levels - 1)

        # Dequantize: map back to float domain
        quantized_x = q_x * scale + x_min

        return quantized_x



    @staticmethod
    def forward(ctx, x: torch.Tensor, n_bits: int):
        ctx.quantize_n_bits = n_bits
        return _QuantizeFunction.quantize(x, n_bits)

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        return _QuantizeFunction.quantize(grad_outputs, ctx.quantize_n_bits), None


class Quantization_Layer(nn.Module):

    def __init__(self, n_bits: int):
        super().__init__()
        self.n_bits = n_bits


    def forward(self, x: torch.Tensor):
        return _QuantizeFunction.apply(x, self.n_bits)





class ClientModel(nn.Module):
    def __init__(self, centralized_base_model, quant_module, sparse_module, device, split_layer, compress_cls_token, num_classes, rand_top_k_sparsity, bit_width, entropy_coding):
        super().__init__()
        self.device = device
        self.split_layer = split_layer

        # --- Deepcopy components from Centralized Model ---
        # 1. Input Processing
        self.patch_size = centralized_base_model.vit.patch_size
        self.hidden_dim = centralized_base_model.vit.hidden_dim
        self.conv_proj = copy.deepcopy(centralized_base_model.vit.conv_proj)
        self.image_size = centralized_base_model.vit.image_size

        # 2. Embeddings & Tokens
        self.class_token = copy.deepcopy(centralized_base_model.vit.class_token)
        self.pos_embedding = copy.deepcopy(centralized_base_model.vit.encoder.pos_embedding)
        self.dropout = copy.deepcopy(centralized_base_model.vit.encoder.dropout)

        # 3. Transformer Blocks (up to split)
        # Note: centralized_model.vit.encoder.layers has [Blocks ... AE ... Blocks]
        # We need everything BEFORE the AE.
        client_layers = []
        for i in range(self.split_layer):
            client_layers.append(centralized_base_model.vit.encoder.layers[i])
        self.blocks = copy.deepcopy(nn.Sequential(*client_layers))

        # 4. AE Encoder

        self.quant_module = copy.deepcopy(quant_module)
        self.sparse_module = copy.deepcopy(sparse_module)


        self.compress_cls_token = compress_cls_token

        self.to(device)

    def forward(self, x):
        # 1. Pre-processing (Client-side logic)
        x = x.to(self.device)

        # Access the underlying ViT components
        n = x.shape[0]

        # Patching and Embedding
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, -1).permute(0, 2, 1)
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")

        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.blocks(x)

        return x

    def compress_decompress(self, x):
        if self.compress_cls_token:
            cls_token = x[:, 0:1, :]
            patches = x[:, 1:, :]
            cls_comm_size_bytes = get_communication_size(cls_token.clone())
        else:
            patches = x

        x_comm_size_bytes = get_communication_size(patches.clone())

        sparse_compr_ratio = 1.0
        quant_compr_ratio = 1.0
        if self.sparse_module:
            shape = patches.shape
            batch_size = shape[0]
            patch_nr = shape[1]

            activations_size = batch_size * patch_nr * self.hidden_dim  # B x L x H

            patches = self.sparse_module.forward(patches)

            compr_ratio = self.sparse_module.get_compression_level(activations_size)
            sparse_compr_ratio = compr_ratio
            x_comm_size_bytes /= compr_ratio

        if self.quant_module:
            quant_compr_ratio = (32.0 / self.quant_module.n_bits)
            x_comm_size_bytes /= quant_compr_ratio  # Adjust for quantization

            # print(f"CLS Token Communication Size (bytes): {cls_comm_size_bytes}")
            # print(f"Patches Communication Size (bytes): {x_comm_size_bytes} \n")


            patches = self.quant_module.forward(patches)

        comm_size_bytes = x_comm_size_bytes
        if self.compress_cls_token:
            x_reconstructed = torch.cat([cls_token, patches], dim=1)
            comm_size_bytes += cls_comm_size_bytes
        else:
            x_reconstructed = patches

        mse = F.mse_loss(x_reconstructed, x).item()

        total_compr_ratio = sparse_compr_ratio * quant_compr_ratio

        if not self.quant_module and not self.sparse_module:
            x_reconstructed = x
            mse = 0.0



        return x_reconstructed, mse, comm_size_bytes

    def switch_to_device(self, device):
        self.to(device)
        return self

class ServerModel(nn.Module):
    def __init__(self, centralized_base_model, device, split_layer, compress_cls_token):
        super().__init__()
        self.device = device

        # Get all encoder layers after the split_layer
        self.server_blocks = nn.Sequential(*list(centralized_base_model.vit.encoder.layers)[split_layer + 1 :])
        self.final_layer_norm = centralized_base_model.vit.encoder.ln
        self.heads = centralized_base_model.vit.heads

        self.compress_cls_token = compress_cls_token



    def forward_uncompressed(self, x):
        # Run remaining blocks
        x = self.server_blocks(x)

        # Final Layer Norm and Head
        x = self.final_layer_norm(x)
        x = x[:, 0]
        x = self.heads(x)

        return x

    def switch_to_device(self, device):
        self.to(device)
        return self