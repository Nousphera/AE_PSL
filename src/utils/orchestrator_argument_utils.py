import argparse
import os
from distutils.util import strtobool

from available_datasets import dataloaders
from models import SupportedModel
from models.auto_encoder import AE_REGISTRY

def set_env_variables(arguments):
    os.environ['TORCH_DATA_DIR'] = arguments['torch_data_dir']
    os.environ['PRE_PROCESSORS_CACHE_DIR'] = arguments['pre_processors_cache_dir']
    os.environ['TOKENIZER_WEIGHTS_CACHE_DIR'] = arguments['tokenizer_weights_cache_dir']
    os.environ['MODEL_WEIGHTS_DIR'] = arguments['model_weights_dir']
    os.environ['AE_WEIGHTS_DIR'] = arguments['ae_weights_dir']


def build_base_argument_parser():
    """
    Builds and returns an instance of ArgumentParser, with all shared parameters across the possible configurations (centralized, split-learning, and federated-learning with and without the adapter approach).
    """
    parser = argparse.ArgumentParser()

    # == Data directories ==
    parser.add_argument('--torch_data_dir', type=str, default='../../shared_data/datasets', help='The directory for the data of all datasets.')
    parser.add_argument('--pre_processors_cache_dir', type=str, default='../../shared_data/preprocessors', help='The directory for all pre-processor weights.')
    parser.add_argument('--tokenizer_weights_cache_dir', type=str, default='../../shared_data/tokenizers', help='The directory for all tokenizer weights.')
    parser.add_argument('--model_weights_dir', type=str, default='../../shared_data/model_checkpoints', help='The directory of all model weights. This is the directory in which the codebase expects the Meta-Transformer model weights to be present.')
    parser.add_argument('--val_mode', nargs='+', type=str, choices=['none', 'early_saving', 'early_stopping'],
                        default='none', help='Whether a validation set should be used. TODO full explanation')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='The proportion of the dataset to use for validation.')

    # == Dataset and model ==
    parser.add_argument('--dataset', nargs='+', type=str, required=True, choices=dataloaders.keys(), help='The dataset that should be used.')

    parser.add_argument('--model', nargs='+', type=str, help='The model that should be used.', choices=[x.lower() for x in SupportedModel.__members__.keys()], default='vit_b_32')

    parser.add_argument('--freeze_clientside', nargs='+', dest='freeze_clientside', type=lambda x: bool(strtobool(x)), default=False, help='Whether all blocks of the encoder should be frozen.')

    parser.add_argument('--split_layer', nargs='+', type=int, default=5, help='Where the model should be split and the AE inserted if applicable')
    # LoRA
    parser.add_argument('--use_lora', nargs='+', dest='use_lora', type=lambda x: bool(strtobool(x)), default=True, help='Whether LoRA should be used to finetune the model.')
    parser.add_argument('--lora_rank', nargs='+', type=int, default=16, help='The rank of the LoRA matrices.')
    parser.add_argument('--lora_alpha', nargs='+', type=int, default=32, help='The alpha scaling factor of LoRA.')

    # == Misc ==
    parser.add_argument('--batch_size', nargs='+', type=int, default=125, help='The batch size on the server. This means that with a batch_size of 500 with 25 clients, each client would have a mini-batch size of 20.')
    parser.add_argument('--nr_of_epochs', nargs='+', type=int, default=10)
    parser.add_argument('--start_lr', nargs='+', type=float, default=1e-4)
    parser.add_argument('--scheduler', nargs='+', dest='scheduler', type=str, default='cosine', help='The learning rate scheduler to use for fine-tuning.', choices=['constant', 'step', 'cosine'])
    parser.add_argument('--random_seed', nargs='+', type=int, default=42)
    parser.add_argument('--single_seed_per_process', type=lambda x: bool(strtobool(x)), default=False)

    parser.add_argument('--num_workers_finetuning', nargs='+', type=int, default=5, help='num_workers provided to the Dataloader, on a per-process basis')
    parser.add_argument('--num_workers_ae_pretraining', nargs='+', type=int, default=4, help='num_workers provided to the Dataloader, on a per-process basis')
    parser.add_argument('--num_workers_activation_loading', nargs='+', type=int, default=16, help='num_workers provided to the Dataloader, on a per-process basis')

    parser.add_argument('--save_model_after_each_epoch',  dest='save_model_after_each_epoch', type=lambda x: bool(strtobool(x)), default=False, help='Whether model(s) weights should be saved after each training epoch.')
    parser.add_argument('--save_final_model', dest='save_final_model', type=lambda x: bool(strtobool(x)), default=False, help='Whether the final obtained model weights after training should be saved.')
    parser.add_argument('--gpu_id', type=str, default=None, help='If desired, this parameter can be used to explicitly define which CUDA gpu to use.')


    # For testing
    parser.add_argument('--small_test_run', dest='small_test_run', type=lambda x: bool(strtobool(x)), default=False, help='Whether a small test run should be executed for rapid testing purposes.')

    parser.add_argument('--encoder_only_client_specific_alignment', nargs='+', dest='encoder_only_client_specific_alignment', type=lambda x: bool(strtobool(x)), default=False, help='Whether the AE should be warmed up on downstream data before starting with the main training loop..')
    parser.add_argument('--concurrent_mse_alignment', nargs='+', dest='concurrent_mse_alignment', type=lambda x: bool(strtobool(x)), default=False, help='Whether the AE should be trained separately during fine-tuning using an MSE loss.')
    parser.add_argument('--client_specific_alignment', nargs='+', dest='client_specific_alignment', type=lambda x: bool(strtobool(x)), default=True, help='Whether the decoders of the AE should be aggregated during AE warmup.')
    parser.add_argument('--iterative_client_specific_alignment_epochs', nargs='+', dest='iterative_client_specific_alignment_epochs', type=int, default=1, help='The number of epochs to use for AE warmup.')
    parser.add_argument('--iterative_client_specific_alignment', nargs='+', dest='iterative_client_specific_alignment', type=lambda x: bool(strtobool(x)), default=False, help='Whether the decoder weights of the AE should be redistributed after an AE warmup epoch. This incurs extra communication overhead, but better aligns training in the next epoch')
    parser.add_argument('--no_compression_batches', nargs='+', dest='no_compression_batches', type=int, default=0, help='The number of initial batches during which no compression should be applied')
    parser.add_argument('--use_compression_during_local_eval', nargs='+', dest='use_compression_during_local_eval', type=lambda x: bool(strtobool(x)), default=True, help='Whether compression should be applied during local evaluation on the clients.')
    parser.add_argument('--use_compression_during_global_eval', nargs='+', dest='use_compression_during_global_eval', type=lambda x: bool(strtobool(x)), default=False, help='Whether compression should be applied during global evaluation')
    parser.add_argument('--compress_cls_token', nargs='+', dest='compress_cls_token', type=lambda x: bool(strtobool(x)), default=False, help='Whether or not to compress the CLS token.')


    parser.add_argument('--plot_distributions', nargs='+', dest='plot_distributions', type=lambda x: bool(strtobool(x)), default=False, help='')

    parser.add_argument('--profile_flops', nargs='+', dest='profile_flops', type=lambda x: bool(strtobool(x)), default=False, help='Whether to profile the number of FLOPs during training.')

    return parser




def expand_argument_parser_with_ae_pretraining_parameters(argument_parser):
    # == For enabling AE pre-training ==
    argument_parser.add_argument('--ae_force_retrain', dest='ae_force_retrain', type=lambda x: bool(strtobool(x)), default=False, help='Whether existing AE weights should be ignored and the AE should be re-trained from scratch.')

    argument_parser.add_argument('--ae_latent_dim', nargs='+', dest='ae_latent_dim',  type=int, default=24, help='The latent dimension of the AE.')
    argument_parser.add_argument('--ae_type', nargs='+', dest='ae_type', type=str, default='2_layer_MLP_(d_d_d_z)', help='The type of AE that should be used.',
                                 choices=AE_REGISTRY)

    argument_parser.add_argument('--ae_pretrain_dataset', nargs='+', dest='ae_pretrain_dataset', type=str, default='imagenet', help='The dataset that should be used for AE pre-training.')
    argument_parser.add_argument('--ae_pretrain_dataset_fraction', nargs='+', dest='ae_pretrain_dataset_fraction', type=float, default=1.0, help='The fraction of the AE pre-training dataset that should be used for AE pre-training.')

    argument_parser.add_argument('--ae_pretrain_epochs', nargs='+', dest='ae_pretrain_epochs', type=int, default=30, help='The number of epochs to use for AE pre-training.')
    argument_parser.add_argument('--ae_pretrain_batch_size', nargs='+', dest='ae_pretrain_batch_size', type=int, default=512, help='The batch size to use for AE pre-training.')
    argument_parser.add_argument('--ae_pretrain_start_lr', nargs='+', dest='ae_pretrain_start_lr', type=float, default=1e-3, help='The starting learning rate to use for AE pre-training.')
    argument_parser.add_argument('--ae_pretrain_optimizer', nargs='+', dest='ae_pretrain_optimizer', type=str, default='adam', help='The optimizer to use for AE pre-training.', choices=['adam', 'adamw', 'sgd'])
    argument_parser.add_argument('--ae_pretrain_scheduler', nargs='+', dest='ae_pretrain_scheduler', type=str, default='cosine', help='The learning rate scheduler to use for AE pre-training.', choices=['step', 'cosine'])
    argument_parser.add_argument('--ae_pretrain_scheduler_step_size', nargs='+', dest='ae_pretrain_scheduler_step_size', type=int, default=5)
    argument_parser.add_argument('--ae_pretrain_loss_fn', nargs='+', dest='ae_pretrain_loss_fn', type=str, default='mse', help='The loss function to use for AE pre-training.', choices=['mse', 'l1'])
    argument_parser.add_argument('--ae_pretrain_cosine_eta_min', nargs='+', dest='ae_pretrain_cosine_eta_min', type=float, default=1e-4, help='The lowest reachable LR during cosine annealing')
    argument_parser.add_argument('--ae_freeze_encoder_during_finetuning', nargs='+', dest='ae_freeze_encoder_during_finetuning', type=lambda x: bool(strtobool(x)), default=True, help='Whether the encoder of the AE should be frozen during downstream fine-tuning.')
    argument_parser.add_argument('--ae_freeze_decoder_during_finetuning', nargs='+', dest='ae_freeze_decoder_during_finetuning', type=lambda x: bool(strtobool(x)), default=True, help='Whether the encoder of the AE should be frozen during downstream fine-tuning.')



    argument_parser.add_argument('--ae_weights_dir', dest='ae_weights_dir', type=str, default='../../data/ae_checkpoints', help='The directory where to scan for existing pre_trained AE weights')
    argument_parser.add_argument('--ae_specific_weights_path', nargs='+', dest='ae_specific_weights_path', type=str, default=None, help='If specified, this AE weights path will be used to load the AE weights, rather than searching in the ae_weights_dir for compatible weights.')
    argument_parser.add_argument('--ae_save_final_weights', dest='ae_save_final_weights', type=lambda x: bool(strtobool(x)), default=True, help='Whether the final AE weights after pre-training should be saved.')
    argument_parser.add_argument('--ae_pretrain_only', dest='ae_pretrain_only', type=lambda x: bool(strtobool(x)), default=False, help='Whether only AE pre-training should be executed, after which the program exits.')
    argument_parser.add_argument('--ae_finetune_lr', nargs='+', dest='ae_finetune_lr', type=float, default=1e-3, help='The learning rate to use when using the AE during DFT -- this is only for the Seperate MSE study in the Appendix')
    argument_parser.add_argument('--ae_general_alignment', nargs='+', dest='ae_general_alignment', type=lambda x: bool(strtobool(x)), default=True, help='Whether General Aligment (GA) should be used. Note: Still need to pass parameters to unfreeze the AE, and disable CSA. This just disables GA ')

    return argument_parser

def expand_argument_parser_with_distributed_learning_parameters(argument_parser):
    argument_parser.add_argument('--nr_of_clients', nargs='+', type=int, default=5, required=False, help='The number of clients to use during training.')
    argument_parser.add_argument('--dataset_split_type', nargs='+', type=str, default='noniid', required=False, help='The type of data distribution that should be used for splitting the original dataset into all separate client-side datasets.')

    return argument_parser


# In src/utils/orchestrator_argument_utils.py

def expand_argument_parser_with_baseline_parameters(argument_parser):
    # Compression/SFL specific arguments

    argument_parser.add_argument('--rand_top_k_sparsity', nargs='+', type=float, default=[0.0], help='Sparsity of Rand-Top-K')
    argument_parser.add_argument('--bit_width', nargs='+', type=int, default=[32], help='Bit width for compressor.')
    argument_parser.add_argument('--adc_compression_ratio', nargs='+', type=float, default=[1], help='Total compression ratio (fraction) for ADC')
    # Add these alongside your existing arguments
    argument_parser.add_argument('--compression_method', nargs='+', type=str, default='ae', choices=['ae', 'rand_top_k', 'adc', 'c3_sl', 'none'],
                        help='The type of intermediate compression to apply.')
    argument_parser.add_argument('--c3_sl_compression_ratio', nargs='+', type=int, default=[1], help='Compression ratio of the C3-SL baseline')





    return argument_parser


def namespace_to_dict(namespace):
    """
    Converts an argparse.Namespace object to a dictionary.
    """
    return {key: value for key, value in vars(namespace).items()}


def namespace_to_global_args_and_search_space(namespace):
    """
    Converts an argparse.Namespace object to a dictionary.
    All values that are lists with more than one element are considered search space variables.
    These are separated from the global arguments,
    and later each element in the search space variables will be used to create different experimental configurations.
    """
    search_space_args = {}
    global_args = {}
    for key, value in vars(namespace).items():
        if isinstance(value, list):
            if len(value) > 1:
                search_space_args[key] = value
            elif len(value) == 1:
                global_args[key] = value[0]
        else:
            global_args[key] = value

    return global_args, search_space_args