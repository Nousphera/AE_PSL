from typing import Any, Tuple

import torch
from models.auto_encoder import BaseViTAE
# from models.fedlite import FedLite_Layer
from models.quantization import Quantization_Layer
from models.random_top_k import RandomTopKModifier


from trainers.implementations.classification.sasl_trainer_heuristic import SASLTrainerHeuristic
from trainers.implementations.classification.sasl_trainer_heuristic_double_attention import SASLTrainerHeuristicDA
from trainers.implementations.classification.sasl_trainer import SASLTrainer
from trainers.implementations.classification.sasl_trainer_heuristic_double_attention_flops import \
    SASLTrainerHeuristicDAFLOPS
from trainers.implementations.classification.sasl_trainer_heuristic_flops import SASLTrainerHeuristicFLOPS
from trainers.implementations.experiment_trainer import ExperimentTrainer

from models.vision_transformer.base.vision_transformer_base import VisionTransformerBase
from utils.model_utils import SupportedModel, get_hidden_dim, get_num_patches

# Centralized dataset configuration to eliminate duplicate mappings
DATASET_CLASS_MAP = {
    'cifar100': 100,
    'food101': 101,
    'sun397': 397,
    'stanford_cars': 196,
    'femnist': 62,
    'audioset': 527,       # AST Dataset
    'speechcommands': 35   # AST Dataset
}


def _get_num_classes(dataset: str) -> int:
    """Helper to safely fetch class count or raise standard error."""
    if dataset not in DATASET_CLASS_MAP:
        raise NotImplementedError(f"Chosen dataset '{dataset}' is not supported.")
    return DATASET_CLASS_MAP[dataset]


# ========= BASE MODEL SELECTION ========= #

def get_base_model(global_args: dict, device: torch.device) -> Any:
    model_name = global_args['model'].upper()
    if model_name not in SupportedModel.__members__:
        raise NotImplementedError(f"Chosen model '{global_args['model']}' is currently not supported.")


    return get_base_model_vit(global_args, device)

    raise NotImplementedError(f"Chosen model '{global_args['model']}' layout logic is missing.")


def get_base_model_vit(global_args: dict, device: torch.device) -> Any:
    num_classes = _get_num_classes(global_args['dataset'])

    from models.vision_transformer.implementations.unimodal.image_classification.models import \
        get_base_model_vit as get_base_model_vit_impl

    return get_base_model_vit_impl(
        vit_type=global_args['model'],
        use_lora=global_args['use_lora'],
        lora_rank=global_args['lora_rank'],
        lora_alpha=global_args['lora_alpha'],
        num_classes=num_classes,
        device=device
    )


# ========= SPLIT MODEL & TRAINER SELECTION ========= #

def get_split_model_pair_and_trainer(
        global_args: dict,
        device: torch.device,
        compress_cls_token: bool,
        base_model: VisionTransformerBase = None,
        auto_encoder: BaseViTAE = None
) -> Tuple[Tuple[Any, Any, bool], ExperimentTrainer]:
    if global_args['model'].upper() not in SupportedModel.__members__:
        raise NotImplementedError('Chosen model is currently not supported.')

    if base_model is None:
        raise ValueError("Base model instance must be provided to get split model architectures.")

    return get_split_model_and_trainer_vit(global_args, base_model, auto_encoder, device, compress_cls_token)


def get_split_model_and_trainer_vit(
        global_args: dict,
        base_model: VisionTransformerBase,
        auto_encoder: BaseViTAE,
        device: torch.device,
        compress_cls_token: bool
) -> Tuple[Tuple[Any, Any, bool], ExperimentTrainer]:
    num_classes = _get_num_classes(global_args['dataset'])
    compression_method = global_args.get('compression_method', 'ae')

    # Routing strategy pattern to clean up the nesting
    if compression_method == 'rand_top_k':
        return _build_rand_top_k_compression(global_args, base_model, num_classes, device)

    elif compression_method == 'adc':
        return _build_adc_compression(global_args, base_model, num_classes, device)

    # elif compression_method == 'fedlite':
    #     return _build_fedlite_compression(global_args, base_model, num_classes, device)

    elif compression_method == 'c3_sl':
        return _build_c3_sl_compression(global_args, base_model, num_classes, device)

    elif compression_method == 'ae':
        return _build_ae_compression(global_args, base_model, auto_encoder, num_classes, device, compress_cls_token)

    else:
        raise ValueError(f"Unsupported compression method: {compression_method}")


# ========= COMPRESSION METHOD BUILDERS (STRATEGIES) ========= #

def _build_c3_sl_compression(global_args: dict, base_model: Any, num_classes: int, device: torch.device):
    from models.c3_sl import ClientModel, ServerModel

    R = global_args['c3_sl_compression_ratio']

    client_model = ClientModel(
        centralized_base_model=base_model,
        split_layer=global_args['split_layer'],
        device=device,
        R=R,
        num_patches=get_num_patches(global_args['model'], global_args) if global_args['compress_cls_token'] else get_num_patches(global_args['model']) - 1,
        hidden_dim=get_hidden_dim(global_args['model']),
        compress_cls_token=global_args['compress_cls_token']
    )

    server_model = ServerModel(
        centralized_base_model=base_model,
        device=device,
        split_layer=global_args['split_layer'],
    )

    trainer = SASLTrainerHeuristicFLOPS() if global_args['profile_flops'] else SASLTrainerHeuristic()

    return (client_model, server_model, True), trainer
def _build_rand_top_k_compression(global_args: dict, base_model: Any, num_classes: int, device: torch.device):
    from models.quantization import ClientModel, ServerModel

    quant_module = Quantization_Layer(n_bits=global_args['bit_width'])
    sparse_module = RandomTopKModifier(rate=global_args['rand_top_k_sparsity'], random_portion=0.1)

    client_model = ClientModel(
        centralized_base_model=base_model,
        quant_module=quant_module,
        sparse_module=sparse_module,
        split_layer=global_args['split_layer'],
        num_classes=num_classes,
        compress_cls_token=global_args['compress_cls_token'],
        device=device,
        rand_top_k_sparsity=global_args['rand_top_k_sparsity'],
        bit_width=global_args['bit_width'],
        entropy_coding=global_args['entropy_coding']
    )

    server_model = ServerModel(
        centralized_base_model=base_model,
        device=device,
        split_layer=global_args['split_layer'],
        compress_cls_token=global_args['compress_cls_token']
    )
    trainer = SASLTrainerHeuristicFLOPS() if global_args['profile_flops'] else SASLTrainerHeuristic()
    return (client_model, server_model, True), trainer

#
# def _build_fedlite_compression(global_args: dict, base_model: Any, num_classes: int, device: torch.device):
#     from models.fedlite import ClientModel, ServerModel
#
#     q = global_args['fedlite_q']
#     r = global_args['fedlite_r']
#     l = global_args['fedlite_l']
#
#     quant_module = FedLite_Layer(q=q, r=r, l=l)
#
#     client_model = ClientModel(
#         centralized_base_model=base_model,
#         quant_module=quant_module,
#         split_layer=global_args['split_layer'],
#         device=device
#     )
#
#     server_model = ServerModel(
#         centralized_base_model=base_model,
#         device=device,
#         split_layer=global_args['split_layer']
#     )
#
#     return (client_model, server_model, True), SASLTrainerHeuristic()


def _build_adc_compression(global_args: dict, base_model: Any, num_classes: int, device: torch.device):
    from models.adc import ClientModel, ServerModel

    compression_ratio = global_args['adc_compression_ratio']
    # Pre-calculate factor components cleanly
    dim_factor = compression_ratio ** 0.5

    client_model = ClientModel(
        centralized_base_model=base_model,
        device=device,
        split_layer=global_args['split_layer'],
        batch_compression=dim_factor,
        token_compression=dim_factor,
        num_classes=num_classes
    )

    server_model = ServerModel(
        centralized_base_model=base_model,
        device=device,
        split_layer=global_args['split_layer']
    )
    trainer = SASLTrainerHeuristicDAFLOPS() if global_args['profile_flops'] else SASLTrainerHeuristicDA()
    return (client_model, server_model, True), trainer


def _build_ae_compression(global_args: dict, base_model: Any, auto_encoder: Any, num_classes: int, device: torch.device,
                          compress_cls_token: bool):
    from models.vision_transformer.implementations.unimodal.image_classification.models import \
        get_split_model as get_split_model_vit

    client_model, server_model, client_model_requires_any_grad = get_split_model_vit(
        base_model=base_model,
        auto_encoder=auto_encoder,
        split_layer=global_args['split_layer'],
        num_classes=num_classes,
        device=device,
        compress_cls_token=compress_cls_token
    )
    from trainers.implementations.classification.sasl_trainer_flops import SASLTrainerFLOPS

    trainer = SASLTrainerFLOPS() if global_args['profile_flops'] else SASLTrainer()
    return (client_model, server_model, client_model_requires_any_grad), trainer