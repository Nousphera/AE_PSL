import math

import torch
import torch.nn as nn
from torch.nn import init


AE_REGISTRY = {}

def register_AE_type(name):
    """Decorator to register a AE class."""
    def decorator(cls):
        AE_REGISTRY[name] = cls
        return cls
    return decorator

def get_AE_type(class_name):
    """Get a AE class from the registry."""
    return AE_REGISTRY.get(class_name, None)

# function that calls from outside
def initialize_AE(global_args, input_dim):
    ae_type = global_args['ae_type']
    if ae_type not in AE_REGISTRY:
        raise ValueError(f"Unknown AE type '{ae_type}'. "
                         f"Available: {list(AE_REGISTRY.keys())}")

    if ae_type == 'identity':
        ae = AE_REGISTRY[ae_type]()
    elif input_dim == global_args['ae_latent_dim']:
        ae = AE_REGISTRY['identity']()
    else:
        ae = AE_REGISTRY[ae_type](input_dim=input_dim, latent_dim=global_args['ae_latent_dim'], global_args=global_args)


    return ae

@register_AE_type('identity')
class BaseViTAE(nn.Module):
    """
    Base class that handles ViT [CLS] token splitting/merging.
    Subclasses only need to define 'self.patch_encoder' and 'self.patch_decoder'.
    """

    def __init__(self):
        super().__init__()
        # Default: Identity (No compression)
        self.patch_encoder = nn.Identity()
        self.patch_decoder = nn.Identity()

    def encode(self, x):
        """
        Splits CLS token and patches. Encodes patches.
        Returns: (compressed_patches, cls_token)
        """
        # x: (Batch, Seq_Len, Dim)
        cls_token = x[:, 0:1, :]
        patches = x[:, 1:, :]

        # Apply specific compression implementation
        z_patches = self.patch_encoder(patches)

        return z_patches, cls_token


    def encode_with_cls(self, x):
        """
        Encodes the entire sequence including CLS token.
        Returns: Compressed sequence (including CLS token)
        """
        return self.patch_encoder(x)

    def decode(self, z_patches, cls_token):
        """
        Decodes patches and recombines with CLS token.
        Returns: Reconstructed sequence
        """
        rec_patches = self.patch_decoder(z_patches)

        # Recombine
        return torch.cat([cls_token, rec_patches], dim=1)

    def decode_with_cls(self, z_seq):
        """
        Decodes the entire compressed sequence (including CLS token).
        Returns: Reconstructed sequence (including CLS token)
        """
        return self.patch_decoder(z_seq)

    def forward(self, x):
        z_patches, cls_token = self.encode(x)
        return self.decode(z_patches, cls_token)

    def forward_with_cls(self, x):
        z_seq = self.encode_with_cls(x)
        return self.decode_with_cls(z_seq)

    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self, freeze_encoder=False, freeze_decoder=False):
        """Freeze encoder and/or decoder parameters."""

        for param in self.patch_encoder.parameters():
            param.requires_grad = not freeze_encoder
        for param in self.patch_decoder.parameters():
            param.requires_grad = not freeze_decoder

    def get_decoder_communication_size(self):
        """
        Computes the size of the patch_decoder parameters in MB.
        """
        # Total number of elements (weights and biases)
        total_params = sum(p.numel() for p in self.patch_decoder.parameters())

        param_list = list(self.patch_decoder.parameters())
        if not param_list:
            return 0.0

        element_size = param_list[0].element_size()  # bytes per element

        size_mb = (total_params * element_size) / (1024 ** 2)

        return size_mb

    def load_weights(self, path):
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict)




@register_AE_type('1_layer_MLP_(d_d_z)')
class single_non_linear(BaseViTAE):
    def __init__(self, input_dim, latent_dim, global_args, dropout=0.01):
        super().__init__()

        # Directly define the patches transformation
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.patch_decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            # Output LayerNorm usually removed in reconstruction to allow variance
        )

        self._init_weights()

    def _init_weights(self):
        # Access layers by index directly
        init.kaiming_uniform_(self.patch_encoder[0].weight, nonlinearity='relu')
        init.xavier_uniform_(self.patch_decoder[0].weight)
        init.zeros_(self.patch_encoder[0].bias)
        init.zeros_(self.patch_decoder[0].bias)

@register_AE_type('1_layer_MLP_(d_d_z)_linear')
class single_linear(BaseViTAE):
    def __init__(self, input_dim, latent_dim, global_args, dropout=0.01):
        super().__init__()

        # Directly define the patches transformation
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout)
        )

        self.patch_decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            # Output LayerNorm usually removed in reconstruction to allow variance
        )

        self._init_weights()

    def _init_weights(self):
        # Access layers by index directly
        init.kaiming_uniform_(self.patch_encoder[0].weight, nonlinearity='relu')
        init.xavier_uniform_(self.patch_decoder[0].weight)
        init.zeros_(self.patch_encoder[0].bias)
        init.zeros_(self.patch_decoder[0].bias)

@register_AE_type('2_layer_MLP_(d_d_4_d_z)')
class double_non_linear(BaseViTAE):
    def __init__(self, input_dim, latent_dim, global_args, dropout=0.01):
        super().__init__()

        intermediate_dim = 192

        # --- ENCODER ---
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, latent_dim),
            # Removed LayerNorm and GELU here to allow negative values in latent space
            nn.Dropout(dropout)
        )

        # --- DECODER ---
        self.patch_decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            # Removed LayerNorm here to allow natural variance in output
            nn.Linear(intermediate_dim, input_dim)
        )

        # --- Initialization ---
        self._init_weights()

    def _init_weights(self):
        # He initialization for the ReLU/GELU layers
        init.kaiming_uniform_(self.patch_encoder[0].weight, nonlinearity='relu')
        init.zeros_(self.patch_encoder[0].bias)

        init.kaiming_uniform_(self.patch_decoder[0].weight, nonlinearity='relu')
        init.zeros_(self.patch_decoder[0].bias)

        # Xavier/Orthogonal is often better for the linear projection layers
        init.xavier_uniform_(self.patch_encoder[3].weight)
        init.zeros_(self.patch_encoder[3].bias)

        init.xavier_uniform_(self.patch_decoder[3].weight)
        init.zeros_(self.patch_decoder[3].bias)

# DEFAULT CHOICE
# 2-layer-MLP-(d-d-d_z)
@register_AE_type('2_layer_MLP_(d_d_d_z)')
class double_non_linear_straight(BaseViTAE):
    def __init__(self, input_dim, latent_dim, global_args, dropout=0.01):
        super().__init__()

        intermediate_dim = input_dim

        # --- ENCODER ---
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, latent_dim),
            # Removed LayerNorm and GELU here to allow negative values in latent space
            nn.Dropout(dropout)
        )

        # --- DECODER ---
        self.patch_decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            # Removed LayerNorm here to allow natural variance in output
            nn.Linear(intermediate_dim, input_dim)
        )

        # --- Initialization ---
        self._init_weights()

    def _init_weights(self):
        # He initialization for the ReLU/GELU layers
        init.kaiming_uniform_(self.patch_encoder[0].weight, nonlinearity='relu')
        init.zeros_(self.patch_encoder[0].bias)

        init.kaiming_uniform_(self.patch_decoder[0].weight, nonlinearity='relu')
        init.zeros_(self.patch_decoder[0].bias)

        # Xavier/Orthogonal is often better for the linear projection layers
        init.xavier_uniform_(self.patch_encoder[3].weight)
        init.zeros_(self.patch_encoder[3].bias)

        init.xavier_uniform_(self.patch_decoder[3].weight)
        init.zeros_(self.patch_decoder[3].bias)

@register_AE_type('2_layer_MLP_(d_d_d_z)_linear')
class double_linear_straight(BaseViTAE):
    def __init__(self, input_dim, latent_dim, global_args, dropout=0.01):
        super().__init__()

        intermediate_dim = input_dim

        # --- ENCODER ---
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, latent_dim),
            # Removed LayerNorm and GELU here to allow negative values in latent space
            nn.Dropout(dropout)
        )

        # --- DECODER ---
        self.patch_decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            # Removed LayerNorm here to allow natural variance in output
            nn.Linear(intermediate_dim, input_dim)
        )

        # --- Initialization ---
        self._init_weights()

    def _init_weights(self):
        # He initialization for the ReLU/GELU layers
        init.kaiming_uniform_(self.patch_encoder[0].weight, nonlinearity='relu')
        init.zeros_(self.patch_encoder[0].bias)

        init.kaiming_uniform_(self.patch_decoder[0].weight, nonlinearity='relu')
        init.zeros_(self.patch_decoder[0].bias)

        # Xavier/Orthogonal is often better for the linear projection layers
        init.xavier_uniform_(self.patch_encoder[2].weight)
        init.zeros_(self.patch_encoder[2].bias)

        init.xavier_uniform_(self.patch_decoder[2].weight)
        init.zeros_(self.patch_decoder[2].bias)

@register_AE_type('2_layer_MLP_(d_d_d_z)_ReLU')
class double_non_linear_straight_ReLU(BaseViTAE):
    def __init__(self, input_dim, latent_dim, global_args, dropout=0.01):
        super().__init__()

        intermediate_dim = input_dim

        # --- ENCODER ---
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim),
            # Removed LayerNorm and GELU here to allow negative values in latent space
            nn.Dropout(dropout)
        )

        # --- DECODER ---
        self.patch_decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.ReLU(),
            # Removed LayerNorm here to allow natural variance in output
            nn.Linear(intermediate_dim, input_dim)
        )

        # --- Initialization ---
        self._init_weights()

    def _init_weights(self):
        # He initialization for the ReLU/GELU layers
        init.kaiming_uniform_(self.patch_encoder[0].weight, nonlinearity='relu')
        init.zeros_(self.patch_encoder[0].bias)

        init.kaiming_uniform_(self.patch_decoder[0].weight, nonlinearity='relu')
        init.zeros_(self.patch_decoder[0].bias)

        # Xavier/Orthogonal is often better for the linear projection layers
        init.xavier_uniform_(self.patch_encoder[3].weight)
        init.zeros_(self.patch_encoder[3].bias)

        init.xavier_uniform_(self.patch_decoder[3].weight)
        init.zeros_(self.patch_decoder[3].bias)

@register_AE_type('2_layer_MLP_(d_d_5_4_d_z)')
class double_non_linear_hourglass(BaseViTAE):
    def __init__(self, input_dim, latent_dim, global_args, dropout=0.01):
        super().__init__()

        intermediate_dim = 1024  # Hourglass shape: Expands to a larger intermediate dimension before compressing to latent_dim

        # --- ENCODER ---
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, latent_dim),
            # Removed LayerNorm and GELU here to allow negative values in latent space
            nn.Dropout(dropout)
        )

        # --- DECODER ---
        self.patch_decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            # Removed LayerNorm here to allow natural variance in output
            nn.Linear(intermediate_dim, input_dim)
        )

        # --- Initialization ---
        self._init_weights()

    def _init_weights(self):
        # He initialization for the ReLU/GELU layers
        init.kaiming_uniform_(self.patch_encoder[0].weight, nonlinearity='relu')
        init.zeros_(self.patch_encoder[0].bias)

        init.kaiming_uniform_(self.patch_decoder[0].weight, nonlinearity='relu')
        init.zeros_(self.patch_decoder[0].bias)

        # Xavier/Orthogonal is often better for the linear projection layers
        init.xavier_uniform_(self.patch_encoder[3].weight)
        init.zeros_(self.patch_encoder[3].bias)

        init.xavier_uniform_(self.patch_decoder[3].weight)
        init.zeros_(self.patch_decoder[3].bias)



@register_AE_type('double_linear')
class double_linear(BaseViTAE):
    def __init__(self, input_dim, latent_dim, global_args, dropout=0.01):
        super().__init__()

        intermediate_dim = 192

        # --- ENCODER ---
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, latent_dim),
            # Removed LayerNorm and GELU here to allow negative values in latent space
            nn.Dropout(dropout)
        )

        # --- DECODER ---
        self.patch_decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            # Removed LayerNorm here to allow natural variance in output
            nn.Linear(intermediate_dim, input_dim)
        )

        # --- Initialization ---
        self._init_weights()

    def _init_weights(self):
        # He initialization for the ReLU/GELU layers
        init.kaiming_uniform_(self.patch_encoder[0].weight, nonlinearity='relu')
        init.zeros_(self.patch_encoder[0].bias)

        init.kaiming_uniform_(self.patch_decoder[0].weight, nonlinearity='relu')
        init.zeros_(self.patch_decoder[0].bias)

        # Xavier/Orthogonal is often better for the linear projection layers
        init.xavier_uniform_(self.patch_encoder[3].weight)
        init.zeros_(self.patch_encoder[3].bias)

        init.xavier_uniform_(self.patch_decoder[3].weight)
        init.zeros_(self.patch_decoder[3].bias)


@register_AE_type('3_layer_MLP_(d_d_2_d_4_d_z)')
class triple_non_linear(BaseViTAE):
    def __init__(self, input_dim, latent_dim, global_args, dropout=0.01):
        super().__init__()

        # Layer dimensions
        dim1 = 384
        dim2 = 192

        # --- ENCODER ---
        # Structure: Input -> 192 -> 48 -> Latent
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.LayerNorm(dim1),
            nn.GELU(),
            nn.Linear(dim1, dim2),
            nn.LayerNorm(dim2),
            nn.GELU(),
            nn.Linear(dim2, latent_dim),
            nn.Dropout(dropout)
        )

        # --- DECODER ---
        # Structure: Latent -> 48 -> 192 -> Input
        self.patch_decoder = nn.Sequential(
            nn.Linear(latent_dim, dim2),
            nn.LayerNorm(dim2),
            nn.GELU(),
            nn.Linear(dim2, dim1),
            nn.LayerNorm(dim1),
            nn.GELU(),
            nn.Linear(dim1, input_dim)
        )

        self._init_weights()

    def _init_weights(self):
        # Indices for Linear layers in Encoder: 0, 3, 6
        # Indices for Linear layers in Decoder: 0, 3, 6

        # --- Encoder Init ---
        init.kaiming_uniform_(self.patch_encoder[0].weight, nonlinearity='relu')  # To 192
        init.kaiming_uniform_(self.patch_encoder[3].weight, nonlinearity='relu')  # To 48
        init.xavier_uniform_(self.patch_encoder[6].weight)  # To Latent

        init.zeros_(self.patch_encoder[0].bias)
        init.zeros_(self.patch_encoder[3].bias)
        init.zeros_(self.patch_encoder[6].bias)

        # --- Decoder Init ---
        init.kaiming_uniform_(self.patch_decoder[0].weight, nonlinearity='relu')  # To 48
        init.kaiming_uniform_(self.patch_decoder[3].weight, nonlinearity='relu')  # To 192
        init.xavier_uniform_(self.patch_decoder[6].weight)  # To Output

        init.zeros_(self.patch_decoder[0].bias)
        init.zeros_(self.patch_decoder[3].bias)
        init.zeros_(self.patch_decoder[6].bias)

# --- 2. Helper Modules for CNNs ---
class PatchesToImage(nn.Module):
    """ (B, L, D) -> (B, D, H, W) """
    def __init__(self, spatial_dim):
        super().__init__()
        self.spatial_dim = spatial_dim

    def forward(self, x):
        b, l, d = x.shape
        return x.transpose(1, 2).reshape(b, d, self.spatial_dim, self.spatial_dim)

class ImageToPatches(nn.Module):
    """ (B, D, H, W) -> (B, L, D) """
    def forward(self, x):
        b, d, h, w = x.shape
        return x.flatten(2).transpose(1, 2)

def get_seq_length_for_model(model_name):
    if model_name == 'vit_b_32':
        return 50
    elif model_name == 'vit_b_16':
        return 197
    else:
        raise ValueError(f"Error initializing AE: Unknown model name '{model_name}' for patch size.")

@register_AE_type('1_layer_CONV')
class ConvSpatialAE(BaseViTAE):
    def __init__(self, input_dim, latent_dim, global_args):
        super().__init__()

        # if latent_dim == 48:
        #     latent_dim = 192 # Reducing of patch dimension already contributes to compression ratio

        # We have B x L x D input
        seq_len = get_seq_length_for_model(global_args['model'])
        # We want to create square out L
        self.spatial_dim = int(math.sqrt(seq_len - 1))  # Exclude CLS token
        # Calculate Intermediate Spatial Dim (Output of Encoder Conv)
        # Formula: floor((h + 2*p - k)/s) + 1
        # For k=3, s=2, p=1: floor((S - 1)/2) + 1
        self.encoded_spatial_dim = int((self.spatial_dim + 2 * 1 - 3) / 2) + 1

        print(
            f"ConvSpatialAE: Input {self.spatial_dim}x{self.spatial_dim} -> Encoded {self.encoded_spatial_dim}x{self.encoded_spatial_dim}")

        # Calculate necessary Output Padding for Decoder to match input size
        # H_out = (H_in - 1)*s - 2*p + k - 1 + op + 1
        # H_out = (Enc - 1)*2 + 1 + op
        # We want H_out == self.spatial_dim
        # op = self.spatial_dim - ((self.encoded_spatial_dim - 1) * 2 + 1)
        output_padding = self.spatial_dim - ((self.encoded_spatial_dim - 1) * 2 + 1)



        # BxLxD -> BxDxHxW -> Conv -> BxD'xH'xW' -> BxL'xD'
        # B x 196 x 768 -> B x 768 x 14 x 14 -> Conv -> B x 192 x 7 x 7 -> B x 49 x 192
        # D' = latent_dim
        # L' = spatial_dim // 2  ( so 7/2=3, 14/2=7)

        self.patch_encoder = nn.Sequential(
            PatchesToImage(self.spatial_dim),
            nn.Conv2d(input_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ImageToPatches()
        )

        self.patch_decoder = nn.Sequential(
            PatchesToImage(self.encoded_spatial_dim),
            nn.ConvTranspose2d(
                latent_dim,
                input_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=output_padding  # Dynamic padding (0 for 7x7, 1 for 14x14)
            ),
            ImageToPatches()
        )

        self._init_weights()

        print(self)

    def _init_weights(self):
        # Index 1 is the Conv layer (after PatchesToImage)
        init.kaiming_uniform_(self.patch_encoder[1].weight, nonlinearity='relu')
        init.xavier_uniform_(self.patch_decoder[1].weight)
        init.zeros_(self.patch_encoder[1].bias)
        init.zeros_(self.patch_decoder[1].bias)



