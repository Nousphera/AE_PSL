import sys

import torch
import torchvision
from torch import nn

from models.lora_layer import LoRALinear
from models.vision_transformer.base.ast_models import ASTModel
from models.vision_transformer.base.ast_wrapper import ASTViTWrapper
from models.vision_transformer.base.dino_wrapper import DINOViTWrapper
from utils.model_utils import SupportedModel


class VisionTransformerBase(nn.Module):
    def __init__(self, vit_type: str, use_lora: bool, lora_rank: int, lora_alpha: int, num_classes: int, device):
        super().__init__()
        self.use_lora = use_lora
        self.num_classes = num_classes


        if vit_type == SupportedModel.VIT_B_16.value:
            print(f"Initializing ViT-B/16 (Mode: {'LoRA' if use_lora else 'Full Fine-Tuning'})...")
            weights = torchvision.models.ViT_B_16_Weights.DEFAULT
            self.vit = torchvision.models.vit_b_16(weights=weights)
        elif vit_type == SupportedModel.VIT_B_32.value:
            print(f"Initializing ViT-B/32 (Mode: {'LoRA' if use_lora else 'Full Fine-Tuning'})...")
            weights = torchvision.models.ViT_B_32_Weights.DEFAULT
            self.vit = torchvision.models.vit_b_32(weights=weights)
        elif vit_type == SupportedModel.DINO_B_16.value:
            print(f"Initializing DINO-B/16 (Mode: {'LoRA' if use_lora else 'Full Fine-Tuning'})...")
            local_utils = sys.modules.pop('utils', None)

            try:
                dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            finally:
                # Restore your local 'utils' module back into memory
                if local_utils is not None:
                    sys.modules['utils'] = local_utils
            self.vit = DINOViTWrapper(dino_model)

        elif vit_type == SupportedModel.DINO_S_16.value:
            print(f"Initializing DINO-S/16 (Mode: {'LoRA' if use_lora else 'Full Fine-Tuning'})...")
            local_utils = sys.modules.pop('utils', None)

            try:
                dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            finally:
                # Restore your local 'utils' module back into memory
                if local_utils is not None:
                    sys.modules['utils'] = local_utils
            self.vit = DINOViTWrapper(dino_model)
        elif vit_type == SupportedModel.DINO_S_16.value:
            print(f"Initializing DINO-S/16 (Mode: {'LoRA' if use_lora else 'Full Fine-Tuning'})...")
            local_utils = sys.modules.pop('utils', None)

            try:
                dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            finally:
                # Restore your local 'utils' module back into memory
                if local_utils is not None:
                    sys.modules['utils'] = local_utils
            self.vit = DINOViTWrapper(dino_model)
        elif vit_type == SupportedModel.SSAST_TINY.value:
            print(f"Initializing SSAST Tiny (Mode: {'LoRA' if use_lora else 'Full Fine-Tuning'})...")

            # Load stored SSAST weights
            # weights_dir = os.environ.get('MODEL_WEIGHTS_DIR', 'models')
            # weight_path = os.path.join(weights_dir, 'SSAST-Tiny-Patch-400.pth')  # Update filename to match yours
            weights_dir = 'data/models/SSAST-Tiny-Patch-400.pth'
            # SpeechCommands V2 requires exactly 100 frames for a 1-second 16kHz clip
            # 16000 samples / 160 hop_length = 100 time frames
            ast_model = ASTModel(
                label_dim=num_classes,
                fshape=16,  # Add this line
                tshape=16,  # Add this line
                fstride=10,
                tstride=10,
                input_fdim=128,
                input_tdim=100,
                model_size='tiny',
                pretrain_stage=False,
                load_pretrained_mdl_path=weights_dir,
                device=device
            )

            # # Load stored SSAST weights
            # weights_dir = os.environ.get('MODEL_WEIGHTS_DIR', 'models')
            # weight_path = os.path.join(weights_dir, 'SSAST-Tiny-Patch-400.pth')  # Update filename to match yours
            #
            # if not os.path.exists(weight_path):
            #     raise FileNotFoundError(f"SSAST weights not found at {weight_path}")
            #
            # sd = torch.load(weight_path, map_location=device)
            # # If the weights were wrapped in DDP, remove the 'module.' prefix
            # if list(sd.keys())[0].startswith('module.'):
            #     sd = {k[7:]: v for k, v in sd.items()}
            #
            # ast_model.load_state_dict(sd, strict=False)
            #
            # # Wrap the model to mimic torchvision ViT attributes
            self.vit = ASTViTWrapper(ast_model)

        else:
            raise ValueError(f"Unsupported ViT type: {vit_type}")



        self.device = device

        # Replace with new head for the desired number of classes
        self.vit.heads = nn.Linear(self.vit.heads[0].in_features, num_classes)

        if self.use_lora:
            # MODE A: LoRA Adaptation
            # 1. Freeze EVERYTHING first
            for param in self.vit.parameters():
                param.requires_grad = False

            # 2. Swap Encoder Linear layers to LoRALinear (adds trainable params)
            self._apply_lora_to_encoder(rank=lora_rank, alpha=lora_alpha)

            # 3. Unfreeze Head (Standard practice)
            for param in self.vit.heads.parameters():
                param.requires_grad = True

        else:
            # MODE B: Regular Fine-Tuning
            # Ensure everything is trainable (default behavior, but being explicit is safe)
            for param in self.vit.parameters():
                param.requires_grad = True

        self.vit.to(device)

    def _apply_lora_to_encoder(self, rank, alpha):
        """Recursively swaps Linear layers for LoRALinear in the encoder."""

        def replace_linear_recursion(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    setattr(module, name, LoRALinear(child, rank, alpha))
                else:
                    replace_linear_recursion(child)

        replace_linear_recursion(self.vit.encoder)

    def retrieve_split_layer_activations(self, x, split_layer: int):

        with torch.no_grad():
            x = x.to(self.device)

            # Fix input shape for AST Model
            if hasattr(self.vit, 'ast') and x.dim() == 4:
                if x.shape[1] == 3:
                    x = x[:, 0:1, :, :]  # Reduce to 1 channel
                if x.shape[2:] != (128, 100):
                    x = torch.nn.functional.interpolate(x, size=(128, 100), mode='bilinear', align_corners=False)

            # 1. Patch Embedding Logic

            # (n, c, h, w) -> (n, hidden_dim, out_h, out_w)
            x = self.vit.conv_proj(x)
            n, hidden_dim, out_h, out_w = x.shape

            # (n, hidden_dim, out_h, out_w) -> (n, hidden_dim, (out_h * out_w))
            x = x.reshape(n, hidden_dim, out_h * out_w)

            # (n, hidden_dim, (out_h * out_w)) -> (n, (out_h * out_w), hidden_dim)
            x = x.permute(0, 2, 1)

            # 2. Add Tokens & Position Embeddings
            batch_class_token = self.vit.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")

            x = x + self.vit.encoder.pos_embedding
            x = self.vit.encoder.dropout(x)

            # 3. Forward Pass through Transformer Encoder up to split_layer
            for i in range(split_layer):
                x = self.vit.encoder.layers[i](x)

        return x


    def get_hidden_dim(self):
        return self.vit.hidden_dim
