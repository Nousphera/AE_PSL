from enum import Enum


class SupportedModel(Enum):
    VIT_B_16 = 'vit_b_16'
    VIT_B_32 = 'vit_b_32'
    DINO_B_16 = 'dino_b_16'
    DINO_S_16 = 'dino_s_16'
    SSAST_TINY = 'ssast_tiny'

def get_hidden_dim(model):
    if model == SupportedModel.VIT_B_16.value:
        return 768
    elif model == SupportedModel.VIT_B_32.value:
        return 768
    elif model == SupportedModel.DINO_B_16.value:
        return 768
    elif model == SupportedModel.DINO_S_16.value:
        return 384
    elif model == SupportedModel.SSAST_TINY.value:
        return 192  # Standard AST base dim
    else:
        raise ValueError(f"Unsupported model to retrieve hidden dim: {model}")

def get_num_patches(model, global_args=None):
    if model == SupportedModel.VIT_B_16.value:
        return 197
    elif model == SupportedModel.VIT_B_32.value:
        return 50
    elif model == SupportedModel.DINO_B_16.value:
        return 197
    elif model == SupportedModel.DINO_S_16.value:
        return 197
    elif model == SupportedModel.SSAST_TINY.value:

        f_dim_patches = (128 - 16) // 10 + 1  # equals 12

        t_dim_frames = 100
        t_dim_patches = (t_dim_frames - 16) // 10 + 1  # equals 9

        num_spatial_patches = f_dim_patches * t_dim_patches  # 12 * 9 = 108

        num_extra_tokens = 2

        return num_spatial_patches + num_extra_tokens  # 110 total sequence length
    else:
        raise ValueError(f"Unsupported model to retrieve hidden dim: {model}")