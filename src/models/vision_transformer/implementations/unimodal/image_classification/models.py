import copy

import torch
from torch import nn


from models.auto_encoder import BaseViTAE
from models.vision_transformer.base.ae_vision_transformer import AEVisionTransformer


from utils.mpsl_utils import client_model_requires_any_grad, get_communication_size

from models.vision_transformer.base.vision_transformer_base import VisionTransformerBase


def get_base_model_vit(vit_type: str, use_lora: bool, lora_rank: int, lora_alpha: int, num_classes: int, device):
    return VisionTransformerBase(vit_type=vit_type,
                                 use_lora=use_lora,
                                 lora_rank=lora_rank,
                                 lora_alpha=lora_alpha,
                                 num_classes=num_classes,
                                 device=device)



def get_split_model(base_model: VisionTransformerBase, auto_encoder: BaseViTAE, split_layer: int, num_classes: int, device, compress_cls_token):
    centralized_base_model = BaseModel(
        base_model=base_model,
        auto_encoder=auto_encoder,
        split_layer=split_layer,
        num_classes=num_classes,
        device=device,
        compress_cls_token=compress_cls_token
    )

    _client_model = ClientModel(centralized_base_model, device=device)

    return _client_model, ServerModel(centralized_base_model, device), client_model_requires_any_grad(_client_model)


class BaseModel(AEVisionTransformer):
    def __init__(self, base_model: VisionTransformerBase, auto_encoder: BaseViTAE, split_layer: int, num_classes: int, device, compress_cls_token):
        super().__init__(
            vit=base_model,
            auto_encoder=auto_encoder,
            split_layer=split_layer,
            num_classes=num_classes,
            device=device,
        )
        self.compress_cls_token = compress_cls_token
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        # Since the AE is embedded within the ViT encoder, we can use the standard forward method
        return self.vit(x)

    def forward_measure_comms(self, x):
        """
        Performs a full forward pass while measuring the size of the
        tensors passed from the Client-side to the Server-side.
        """
        # 1. Pre-processing (Client-side logic)
        x = x.to(self.device)

        # Access the underlying ViT components
        vit = self.vit
        n = x.shape[0]
        batch_size = x.shape[0]

        # Patching and Embedding
        x = vit.conv_proj(x)
        x = x.reshape(n, vit.hidden_dim, -1).permute(0, 2, 1)
        batch_class_token = vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + vit.encoder.pos_embedding
        x = vit.encoder.dropout(x)

        # 2. Run blocks before the split
        for i in range(self.split_layer):
            x = vit.encoder.layers[i](x)

        # 3. Run AE Encoder (The Split Point)
        ae_module = vit.encoder.layers[self.split_layer]
        x_latent, cls_token = ae_module.encode(x)

        # --- MEASURE COMMUNICATION SIZE ---
        # We measure both the latent spatial features and the CLS token
        cls_comm_size_bytes = get_communication_size(cls_token.clone()) / batch_size
        x_comm_size_bytes = get_communication_size(x_latent.clone()) / batch_size

        comm_size_bytes = (cls_comm_size_bytes + x_comm_size_bytes)
        # comm_size_bytes = (get_communication_size(x_latent)) / batch_size
        # ----------------------------------

        # 4. Run AE Decoder (Server-side logic)
        x = ae_module.decode(x_latent, cls_token)

        # 5. Run remaining blocks
        for i in range(self.split_layer + 1, len(vit.encoder.layers)):
            x = vit.encoder.layers[i](x)

        # 6. Final Head
        x = vit.encoder.ln(x)
        x = x[:, 0]
        x = vit.heads(x)

        return x, comm_size_bytes

    def switch_to_device(self, device):
        self.to(device)
        return self

    def freeze_encoder(self):
        for param  in self.vit.conv_proj.parameters():
            param.requires_grad = False

        for layer in self.vit.encoder.layers[:self.split_layer]:
            for param in layer.parameters():
                param.requires_grad = False

        for param in self.vit.encoder.layers[self.split_layer].patch_encoder.parameters():
            param.requires_grad = False


class ClientModel(nn.Module):
    def __init__(self, centralized_base_model, device):
        super().__init__()
        self.device = device
        self.split_layer = centralized_base_model.split_layer

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
        # The AE is located exactly at the split_layer index
        # TODO need to only copy the patch encoder portion of the AE module, not the entire module (which includes the decoder)
        self.ae_module = copy.deepcopy(centralized_base_model.vit.encoder.layers[self.split_layer])
        # self.ae_encoder = copy.deepcopy(ae_module.patch_encoder)

        self.compress_cls_token = centralized_base_model.compress_cls_token

        self.to(device)


    def forward(self, x):

        # 1. Pre-processing (Client-side logic)
        x = x.to(self.device)

        # --- AUDIO/AST FIX ---
        if self.conv_proj.in_channels == 1:
            if x.dim() == 3:
                x = x.unsqueeze(1)
            if x.dim() == 4 and x.shape[1] == 3:
                x = x[:, 0:1, :, :]
            if x.shape[3] == 128:
                x = x.transpose(2, 3)
            if x.shape[2:] != (128, 100):
                x = torch.nn.functional.interpolate(x, size=(128, 100), mode='bilinear', align_corners=False)

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


    def compress(self, x):

        # 3. Run AE Encoder (The Split Point)
        if self.compress_cls_token:
            x_latent = self.ae_module.encode_with_cls(x)
            cls_token = torch.empty((x.shape[0], 0, self.hidden_dim), device=x.device, requires_grad=True)
        else:
            x_latent, cls_token = self.ae_module.encode(x)

        # Calculate size for the unified interface
        cls_comm_size_bytes = get_communication_size(cls_token.clone())
        x_comm_size_bytes = get_communication_size(x_latent.clone())
        comm_size_bytes = cls_comm_size_bytes + x_comm_size_bytes

        # batch_size = intermediate_activations.shape[0]
        #
        # # = Communication tracking =
        # # activations_size = get_communication_size(intermediate_activations)
        #
        # patch_nr = intermediate_activations.shape[1]
        # latent_dim = intermediate_activations.shape[-1]
        # float_size = 4  # 4 bytes for a float32
        #
        # activations_size = batch_size * patch_nr * latent_dim * float_size  # B x L x latent x 4 bytes
        # cls_size = batch_size * 1 * hidden_dim * float_size  # B x 1 x hidden x 4 bytes (assuming CLS token has same hidden dimension as patch tokens)
        # comms_size = activations_size + cls_size

        return x_latent, cls_token, comm_size_bytes

    def switch_to_device(self, device):
        self.to(device)
        return self


    def set_lora_trainable(self, trainable: bool = True):
        """
        Toggles the grad status for LoRA parameters.
        Ensures the ViT backbone, embeddings, and AE remain frozen.
        """
        for name, param in self.named_parameters():
            # Specifically target the names defined in your LoRALinear class
            if "lora_a" in name or "lora_b" in name:
                param.requires_grad = trainable
            else:
                # Force everything else to stay frozen (Backbone, AE, Embeddings)
                param.requires_grad = False


class ServerModel(nn.Module):
    def __init__(self, centralized_base_model, device):
        super().__init__()
        self.device = device

        # Server model is only replicated a single time, so we don't need to deepcopy
        self.ae_module = centralized_base_model.vit.encoder.layers[centralized_base_model.split_layer]
        # Get all encoder layers after the split_layer
        self.server_blocks = nn.Sequential(*list(centralized_base_model.vit.encoder.layers)[centralized_base_model.split_layer + 1 :])
        self.final_layer_norm = centralized_base_model.vit.encoder.ln
        self.heads = centralized_base_model.vit.heads

        self.compress_cls_token = centralized_base_model.compress_cls_token

    # def forward(self, x, cls):
    #     # x = x.to(self.device)
    #
    #     # 4. Run AE Decoder (Server-side logic)
    #     x = self.ae_module.decode(x, cls)
    #
    #     # 5. Run remaining blocks
    #     x = self.server_blocks(x)
    #
    #     # 6. Final Head
    #     x = self.final_layer_norm(x)
    #     x = x[:, 0]
    #     x = self.heads(x)
    #
    #     return x

    def decompress(self, x, cls):
        # Run AE Decoder (Server-side logic)

        if self.compress_cls_token:
            # In this case the CLS token is empty
            x = self.ae_module.decode_with_cls(x)
        else:
            x = self.ae_module.decode(x, cls)

        return x

    def forward_uncompressed(self, x):
        x = self.server_blocks(x)
        x = self.final_layer_norm(x)

        # Check if the model has a dist_token (AST/DeiT behavior)
        if x.shape[1] > 1 and hasattr(self.heads, 'mlp_head') or "AST" in str(type(self)):
            # AST averages the CLS and DIST tokens
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            # Standard ViT behavior
            x = x[:, 0]

        x = self.heads(x)
        return x

    def switch_to_device(self, device):
        self.to(device)
        return self

