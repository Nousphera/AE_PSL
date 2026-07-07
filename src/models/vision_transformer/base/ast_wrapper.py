import torch
from torch import nn


class ASTEncoderWrapper(nn.Module):
    """
    Wraps the internal timm blocks to match the expected 'encoder' structure
    used by ClientModel and VisionTransformerBase.
    """

    def __init__(self, ast_model):
        super().__init__()
        self.ast = ast_model
        # Map AST's timm pos_embed and dropout
        self.pos_embedding = self.ast.v.pos_embed
        self.dropout = self.ast.v.pos_drop

        # Map the transformer blocks
        self.layers = self.ast.v.blocks

        # Map the final layer norm
        self.ln = self.ast.v.norm


class ASTViTWrapper(nn.Module):
    """
    Wraps the ASTModel to mimic the standard torchvision ViT attribute structure.
    """

    def __init__(self, ast_model):
        super().__init__()
        self.ast = ast_model

        # Map basic dimensions
        self.patch_size = self.ast.v.patch_embed.patch_size
        self.image_size = self.ast.v.patch_embed.img_size  # Note: This is a tuple in AST
        self.hidden_dim = self.ast.original_embedding_dim

        # Map the patch embedding projection layer
        self.conv_proj = self.ast.v.patch_embed.proj

        # CRITICAL FIX: AST uses DeiT, which has both a cls_token and a dist_token.
        # We concatenate them here so that your hardcoded ClientModel logic:
        # `x = torch.cat([batch_class_token, x], dim=1)`
        # naturally injects BOTH tokens into the sequence.
        self.class_token = nn.Parameter(
            torch.cat([self.ast.v.cls_token, self.ast.v.dist_token], dim=1)
        )

        # Wrap the encoder components
        self.encoder = ASTEncoderWrapper(self.ast)

    @property
    def heads(self):
        """
        Mimics the `heads` attribute of standard torchvision ViT.
        Standard ViT has self.heads as an nn.Sequential with the linear layer at index 0.
        ASTModel uses an mlp_head defined as nn.Sequential(nn.LayerNorm, nn.Linear).
        We return an nn.Sequential containing just the Linear layer so that
        `self.vit.heads[0].in_features` works properly.
        """
        # If it's the original structure (LayerNorm, Linear)
        if isinstance(self.ast.mlp_head, nn.Sequential) and len(self.ast.mlp_head) == 2 and isinstance(
                self.ast.mlp_head[0], nn.LayerNorm):
            return nn.Sequential(self.ast.mlp_head[1])
        # Fallback for when it has been overwritten or structured differently
        elif isinstance(self.ast.mlp_head, nn.Sequential) and len(self.ast.mlp_head) > 0 and isinstance(
                self.ast.mlp_head[-1], nn.Linear):
            return nn.Sequential(self.ast.mlp_head[-1])

        return nn.Sequential(self.ast.mlp_head)

    @heads.setter
    def heads(self, new_head):
        """
        Intercepts assignment to `self.vit.heads`.
        VisionTransformerBase completely overwrites the head with `nn.Linear(...)`.
        We must inject this new head back into the original ASTModel's `mlp_head`,
        preserving its initial LayerNorm.
        """
        if isinstance(self.ast.mlp_head, nn.Sequential) and len(self.ast.mlp_head) >= 1 and isinstance(
                self.ast.mlp_head[0], nn.LayerNorm):
            self.ast.mlp_head = nn.Sequential(
                self.ast.mlp_head[0],  # Preserve original LayerNorm
                new_head
            )
        else:
            self.ast.mlp_head = nn.Sequential(new_head)

    def forward(self, x):
        # Intercept and format the input for AST expectations
        if x.dim() == 4:
            if x.shape[1] == 3:
                x = x[:, 0:1, :, :]
            if x.shape[2:] != (128, 100):
                x = torch.nn.functional.interpolate(x, size=(128, 100), mode='bilinear', align_corners=False)

            # ASTModel expects (batch_size, time_frame_num, frequency_bins)
            # x is currently (N, 1, 128, 100)
            x = x.squeeze(1).transpose(1, 2)  # Becomes (N, 100, 128)

        # We route standard forward passes back to the original AST forward method
        # 'ft_avgtok' evaluates the model using the mean of all tokens for the classification head
        return self.ast(x, task='ft_avgtok')