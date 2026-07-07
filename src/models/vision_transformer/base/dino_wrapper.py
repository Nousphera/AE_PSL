import torch.nn as nn


class DINOEncoderWrapper(nn.Module):
    """Mocks the torchvision vit.encoder structure for the provided DINO specification."""

    def __init__(self, dino_model):
        super().__init__()
        self.dino = dino_model

    @property
    def layers(self):
        # Maps to self.blocks (nn.ModuleList of Block objects)
        return self.dino.blocks

    def __setattr__(self, name, value):
        # Explicitly catch assignments to 'layers' to override the DINO blocks.
        # This prevents PyTorch from silently bypassing property setters.
        if name == 'layers':
            self.dino.blocks = value
        else:
            super().__setattr__(name, value)

    @property
    def pos_embedding(self):
        return self.dino.pos_embed

    @property
    def dropout(self):
        return self.dino.pos_drop

    @property
    def ln(self):
        return self.dino.norm


class DINOViTWrapper(nn.Module):
    """Mocks the top-level torchvision ViT attributes for the provided DINO specification."""

    def __init__(self, dino_model):
        super().__init__()
        self.dino = dino_model
        self.encoder = DINOEncoderWrapper(dino_model)

        # Mocking the torchvision Sequential head to prevent index errors
        # when `self.vit.heads[0].in_features` is called in VisionTransformerBase.
        in_features = self.dino.embed_dim
        self.dino.head = nn.Sequential(nn.Linear(in_features, in_features))

    @property
    def conv_proj(self):
        return self.dino.patch_embed.proj

    @property
    def class_token(self):
        return self.dino.cls_token

    @property
    def hidden_dim(self):
        return self.dino.embed_dim

    @property
    def patch_size(self):
        return self.dino.patch_embed.patch_size

    @property
    def image_size(self):
        return self.dino.patch_embed.img_size

    @property
    def heads(self):
        return self.dino.head

    def __setattr__(self, name, value):
        # Explicitly catch assignments to 'heads' for the exact same PyTorch reason.
        if name == 'heads':
            self.dino.head = value
        else:
            super().__setattr__(name, value)

    def forward(self, x):
        return self.dino(x)