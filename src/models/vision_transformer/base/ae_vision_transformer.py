from models.auto_encoder import BaseViTAE
from models.vision_transformer.base.vision_transformer_base import VisionTransformerBase
from torch import nn

# Vision Transformer with Auto-Encoder Integration.
class AEVisionTransformer(nn.Module):
    def __init__(self, vit: VisionTransformerBase, auto_encoder: BaseViTAE, split_layer: int, num_classes: int, device):
        super().__init__()

        self.vit = vit.vit
        self.split_layer = split_layer
        self.num_classes = num_classes
        self.device = device


        # 3. Insert AE into the encoder sequence
        encoder_layers = list(self.vit.encoder.layers)

        if split_layer < 0 or split_layer > len(encoder_layers):
            raise ValueError(f"Split layer must be between 0 and {len(encoder_layers)}")

        if auto_encoder is None:
            raise ValueError("Auto-encoder instance must be provided to AEVisionTransformer constructor.")

        encoder_layers.insert(split_layer, auto_encoder)
        self.vit.encoder.layers = nn.Sequential(*encoder_layers)

    def forward_full(self, x):
        return self.vit(x)


    def print_status(self):
        """Helper to inspect the model state."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())

        mode = "LoRA" if self.use_lora else "Full Fine-Tuning"
        print(f"\n--- Model Status: {mode} ---")
        print(f"Total Params: {all_params:,}")
        print(f"Trainable Params: {trainable_params:,}")
        print(f"Trainable Ratio: {100 * trainable_params / all_params:.2f}%")

        # Check first layer type
        first_layer = self.vit.encoder.layers[0].mlp[0]
        print(f"Encoder Layer Type: {type(first_layer).__name__}")

# Vision Transformer with Auto-Encoder Integration.
class HeuristicVisionTransformer(nn.Module):
    def __init__(self, vit: VisionTransformerBase, compr_layer: nn.Module, split_layer: int, num_classes: int, device):
        super().__init__()

        self.vit = vit.vit
        self.split_layer = split_layer
        self.num_classes = num_classes
        self.device = device


        # 3. Insert AE into the encoder sequence
        encoder_layers = list(self.vit.encoder.layers)

        if split_layer < 0 or split_layer > len(encoder_layers):
            raise ValueError(f"Split layer must be between 0 and {len(encoder_layers)}")

        if compr_layer is None:
            raise ValueError("Auto-encoder instance must be provided to AEVisionTransformer constructor.")

        encoder_layers.insert(split_layer, compr_layer)
        self.vit.encoder.layers = nn.Sequential(*encoder_layers)

    def forward_full(self, x):
        return self.vit(x)


    def print_status(self):
        """Helper to inspect the model state."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())

        mode = "LoRA" if self.use_lora else "Full Fine-Tuning"
        print(f"\n--- Model Status: {mode} ---")
        print(f"Total Params: {all_params:,}")
        print(f"Trainable Params: {trainable_params:,}")
        print(f"Trainable Ratio: {100 * trainable_params / all_params:.2f}%")

        # Check first layer type
        first_layer = self.vit.encoder.layers[0].mlp[0]
        print(f"Encoder Layer Type: {type(first_layer).__name__}")