import copy
import time
from argparse import Namespace

import torch

import available_datasets as datasets
import models
from ae_trainers.ae_trainer import get_auto_encoder
from available_datasets import get_vit_mnist_transforms
from models import get_base_model
from trainers.implementations.classification.sasl_trainer import SASLTrainer
from trainers.implementations.experiment_results import ExperimentResults
from utils.orchestrator_argument_utils import build_base_argument_parser, \
    expand_argument_parser_with_distributed_learning_parameters, set_env_variables, \
    namespace_to_dict
from utils.config_utils import set_random_seed
from utils.cuda_utils import get_device
from utils.dataloader_utils import HFToTupleDataset, ClientAwareDataset, FederatedBatchSampler, \
    federated_collate_fn
from utils.model_saving_utils import save_split_model, save_experiment_results, \
    load_split_model, delete_model, save_combined_experiment_results
from utils.mpsl_utils import compute_mini_batch_size
from utils.scheduler_utils import get_optimizer_and_scheduler, get_optimzer_and_scheduler_for_seperate_ae_finetuning
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner




def setup_arguments() -> dict:
    parser = build_base_argument_parser()
    parser = expand_argument_parser_with_distributed_learning_parameters(parser)

    parser.add_argument('--test_num_workers', type=int, default=5,
                        help='num_workers provided to the test Dataloader. For Split Learning, we differentiate between num_workers for the train Dataloader and test_num_workers for the test Dataloader.')

    args: Namespace = parser.parse_args()
    set_env_variables(args)
    return namespace_to_dict(args)



def finetune_mpsl(global_args, device, server_model, client_models, trainer,
                  global_train_dataloader, val_dataloaders, test_dataloaders,
                  client_optimizers_main, client_schedulers_main, client_optimizers_ae, client_schedulers_ae,
                  server_optimizer, server_scheduler,
                  client_model_requires_any_grad, max_nr_of_batches_in_epoch,
                  search_space_args, validation_mode, early_stopping_patience=3):
    """
    Handles the fine-tuning loop with support for early stopping and saving.
    """

    # Initialize ExperimentResults
    experiment_results = ExperimentResults(validation_mode=validation_mode)
    experiment_results.params = search_space_args

    # Validation State Variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Initialize the aggregated model variable (using the first client as a template)
    aggregated_client_model = client_models[0]

    use_heuristic = not global_args['compression_method'] == 'ae'

    if global_args['encoder_only_client_specific_alignment'] and global_args['client_specific_alignment']:
        raise ValueError("Can't do both encoder_only_client_specific_alignment and client_specific_alignment at the same time.")


    if global_args['encoder_only_client_specific_alignment'] and not global_args['ae_type'] == 'identity' and not use_heuristic:
        print("Warmup client-side AEs with one pass through the data")

        SASLTrainer().warmup_epoch(
            device=device,
            server_model=server_model,
            server_optimizer=server_optimizer,
            global_train_dataloader=global_train_dataloader,  # <-- Pass the dataloader directly
            client_models=client_models,
            client_model_requires_any_grad=client_model_requires_any_grad,
            client_optimizers_main=client_optimizers_main,
            client_schedulers_main=client_schedulers_main,
            client_optimizers_ae=client_optimizers_ae,
            client_schedulers_ae=client_schedulers_ae,
            max_nr_of_batches_in_epoch=max_nr_of_batches_in_epoch,
            experiment_results=experiment_results,
            epoch_nr=-1,
            global_args=global_args
        )

    elif global_args['client_specific_alignment'] and not global_args['ae_type'] == 'identity' and not use_heuristic:
        print("Warmup client-side AEs with one pass through the data and aggregate decoders before fine-tuning")

        warmup_epochs = global_args['iterative_client_specific_alignment_epochs']


        for epoch in range(global_args['iterative_client_specific_alignment_epochs']):
            # use negative epoch numbers for warmup to differentiate them from the main fine-tuning epochs in logging and results
            current_warmup_epoch = - warmup_epochs + epoch

            SASLTrainer().client_specific_alignment(
                device=device,
                server_model=server_model,
                server_optimizer=server_optimizer,
                global_train_dataloader=global_train_dataloader,  # <-- Pass the dataloader directly
                client_models=client_models,
                client_model_requires_any_grad=client_model_requires_any_grad,
                client_optimizers_main=client_optimizers_main,
                client_schedulers_main=client_schedulers_main,
                client_optimizers_ae=client_optimizers_ae,
                client_schedulers_ae=client_schedulers_ae,
                max_nr_of_batches_in_epoch=max_nr_of_batches_in_epoch,
                experiment_results=experiment_results,
                epoch_nr=current_warmup_epoch,
                global_args=global_args
            )


    print("\nStarting MPSL Fine-tuning\n")


    if len(val_dataloaders) == 0:
        val_dataloaders = None


    use_gpu_transform = global_args['dataset'] == 'cifar100'

    for epoch_nr in range(global_args['nr_of_epochs']):

        # --- Training ---
        train_loss, train_acc, nr_of_elements_per_client_dict = trainer.train_epoch(
            device=device,
            server_model=server_model,
            server_optimizer=server_optimizer,
            global_train_dataloader=global_train_dataloader, # <-- Pass the dataloader directly
            client_models=client_models,
            client_model_requires_any_grad=client_model_requires_any_grad,
            client_optimizers_main=client_optimizers_main,
            client_schedulers_main=client_schedulers_main,
            client_optimizers_ae=client_optimizers_ae,
            client_schedulers_ae=client_schedulers_ae,
            max_nr_of_batches_in_epoch=max_nr_of_batches_in_epoch,
            experiment_results=experiment_results,
            epoch_nr=epoch_nr,
            global_args=global_args
        )

        print(f'Epoch {epoch_nr} --- Train --- Loss: {train_loss:.3f} - Acc: {train_acc * 100.0:.2f}%')

        server_scheduler.step()

        # # --- Aggregation ---
        # aggregated_client_model = fed_avg(client_models,
        #                                   get_client_weight_multipliers__nr_of_elements(nr_of_elements_per_client_dict))
        #
        # # --- Regular Checkpointing ---
        # if global_args['save_model_after_each_epoch']:
        #     save_split_model(aggregated_client_model, server_model, global_args['save_file_name'])

        # --- Validation & Testing ---
        with torch.no_grad():

            # Option 1: Early Saving / Stopping using Validation Set
            if val_dataloaders is not None and validation_mode in ['early_saving', 'early_stopping']:

                val_loss, val_acc, _ = trainer.test_epoch_distributed(client_models, server_model, val_dataloaders, device, experiment_results, epoch_nr, use_gpu_transform, global_args['use_compression_during_local_eval'])


                print(f"Epoch {epoch_nr} --- Validation --- Loss: {val_loss:.3f} - Acc: {val_acc * 100.0:.2f}%")

                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # print(f"Validation loss improved to {best_val_loss:.6f}. Saving model...")
                    print("Validation not working currently -- not saving intermediate model")
                    # save_split_model(aggregated_client_model, server_model, global_args['save_file_name'])
                else:
                    patience_counter += 1
                    if validation_mode == 'early_stopping':
                        print(
                            f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")

                if validation_mode == 'early_stopping' and patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch_nr} epochs.")
                    break

            # Option 2: No Validation (Legacy / Standard Behavior)
            elif validation_mode == 'none':
                test_loss, test_acc, client_results = trainer.test_epoch_distributed(client_models, server_model, test_dataloaders, device,
                                                       experiment_results, epoch_nr,
                                                       use_gpu_transform, global_args['use_compression_during_local_eval'])

                # IMPORTANT: Update final stats during the loop for non-validation mode
                experiment_results.final_test_loss = test_loss
                experiment_results.final_test_accuracy = test_acc
                experiment_results.final_distributed_eval = client_results

                print(f"Epoch {epoch_nr} --- Test --- Loss: {test_loss:.3f} - Acc: {test_acc * 100.0:.2f}%")

        save_experiment_results(experiment_results, global_args['save_file_name'])

    print(f'Finished training phase.')

    # --- Final Test Evaluation ---
    # If early saving/stopping was used, reload the best model
    if validation_mode in ['early_saving', 'early_stopping']:
        print("Loading best model for final testing...")
        try:
            # Load weights into the aggregated model and server model
            print("Validation not working for personalized, using the model of the last epoch")
            load_split_model(aggregated_client_model, server_model, device, global_args['save_file_name'])
        except FileNotFoundError:
            print("Warning: Best model file not found. Using current weights.")

    print('Starting Final Test Set Evaluation')
    with torch.no_grad():
        final_test_loss, final_test_acc, client_results = trainer.test_epoch_distributed(client_models, server_model, test_dataloaders, device,
                                                 None, global_args['nr_of_epochs'],
                                                 use_gpu_transform, global_args['use_compression_during_local_eval'])
        experiment_results.final_test_loss = final_test_loss
        experiment_results.final_test_accuracy = final_test_acc
        experiment_results.final_distributed_eval = client_results
        print(f"Final Test Metrics --- Loss: {final_test_loss:.3f} - Acc: {final_test_acc * 100.0:.2f}%")

    # --- Final Model Management ---
    if validation_mode in ['early_saving', 'early_stopping']:
        if not global_args['save_final_model']:
            # Delete the intermediate best model if user didn't request final model save
            delete_model(global_args['save_file_name'])
    else:
        if global_args['save_final_model']:
            save_split_model(aggregated_client_model, server_model, global_args['save_file_name'])

    return experiment_results


def run_2_stage_mpsl_personalized(global_args: dict, search_space_args: dict):
    set_random_seed(global_args['random_seed'])
    set_env_variables(global_args)

    # # # # # # # # # # # # # # # # # Setup # # # # # # # # # # # # # # # # #
    device = get_device(global_args)
    mini_batch_size = compute_mini_batch_size(global_args['batch_size'], global_args['nr_of_clients'])
    total_batch_size = global_args['nr_of_clients'] * mini_batch_size

    print(f'Device: {device}')
    print(
        f'Available datasets: {datasets.available_datasets()}, chosen: {global_args["dataset"]} with batch_size: {global_args["batch_size"]} and mini-batch size: {mini_batch_size}. Hence real total batch_size is {total_batch_size}')

    if total_batch_size > global_args['batch_size']:
        raise Exception(f'total_batch_size > chosen batch_size ({total_batch_size} > {global_args["batch_size"]})')

    validation_mode = global_args['val_mode']
    validation_split = global_args['val_split']
    if validation_mode == 'none': validation_split = 0.0

    # full_dataset = datasets.load_data(name=global_args['dataset'], num_partitions=global_args['nr_of_clients'],
    #                                   split=global_args['dataset_split_type'], seed=global_args['random_seed'],
    #                                   global_args=global_args, val_split=validation_split)
    #
    # val_dataset = full_dataset.load_validation_set()
    # test_dataset = full_dataset.load_test_set()
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NaturalIdPartitioner(partition_by="writer_id")},
        # dataset_kwargs={"cache_dir": os.environ["TORCH_DATA_DIR"]}  # This points to your specific path
    )

    import os
    from datasets import concatenate_datasets, load_from_disk
    from torch.utils.data import DataLoader, ConcatDataset

    # 1. Configuration & Cache Naming
    total_partitions = fds.partitioners["train"].num_partitions
    nr_of_clients = global_args['nr_of_clients']
    subsample_size = 0.1  # 10% of each partition
    seed = global_args['random_seed']

    # Create a unique cache fingerprint
    cache_dir_name = "fedmnist_dataset_cache"
    cache_dir = os.path.join(os.environ['TORCH_DATA_DIR'], cache_dir_name, f"clients_{nr_of_clients}_sub_{subsample_size}_seed_{seed}")

    partitions_per_client = total_partitions // nr_of_clients

    train_datasets_list = []
    partition_map = {}
    current_offset = 0

    client_val_dataloaders = {}
    client_test_dataloaders = {}

    # Reusable transforms
    train_transform = get_vit_mnist_transforms()
    test_transform = get_vit_mnist_transforms()

    if os.path.exists(cache_dir):
        print(f"Cache found! Loading datasets directly from: {cache_dir}")

        for client_id in range(nr_of_clients):
            client_dir = os.path.join(cache_dir, f"client_{client_id}")

            # Load from disk
            client_train_hf = load_from_disk(os.path.join(client_dir, "train"))
            client_test_hf = load_from_disk(os.path.join(client_dir, "test"))

            if validation_split > 0:
                client_val_hf = load_from_disk(os.path.join(client_dir, "val"))
            else:
                client_val_hf = None

            # Convert HF to PyTorch Datasets
            client_train_ds = HFToTupleDataset(client_train_hf, transform=train_transform)
            client_test_ds = HFToTupleDataset(client_test_hf, transform=test_transform)

            # Save testing loaders
            client_test_dataloaders[client_id] = DataLoader(client_test_ds, batch_size=total_batch_size, shuffle=False)
            if client_val_hf:
                client_val_ds = HFToTupleDataset(client_val_hf, transform=test_transform)
                client_val_dataloaders[client_id] = DataLoader(client_val_ds, batch_size=total_batch_size,
                                                               shuffle=False)

            # Build partition map
            train_datasets_list.append(client_train_ds)
            num_items = len(client_train_ds)
            partition_map[client_id] = list(range(current_offset, current_offset + num_items))
            current_offset += num_items

    else:
        print(f"No cache found. Processing partitions and saving to: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)

        for client_id in range(nr_of_clients):
            start_p_id = client_id * partitions_per_client
            end_p_id = start_p_id + partitions_per_client

            client_train_partitions = []
            client_test_partitions = []

            # Load, subsample, and split EACH partition
            for p_id in range(start_p_id, end_p_id):
                partition_hf = fds.load_partition(partition_id=p_id, split="train")
                num_samples = len(partition_hf)

                # Skip partitions that are impossibly small
                if num_samples < 2:
                    # Need at least 2 samples to have a train and a test split
                    continue

                # enforce a minimum of 2 samples
                target_subsample_size = max(2, int(num_samples * subsample_size))

                if num_samples > target_subsample_size:
                    # Use train_size as an integer absolute value instead of a float ratio
                    subsampled_hf = partition_hf.train_test_split(
                        train_size=target_subsample_size,
                        seed=seed
                    )['train']
                else:
                    subsampled_hf = partition_hf

                # Safely split into train and test: ensure test gets at least 1
                num_subsampled = len(subsampled_hf)
                target_test_size = max(1, int(num_subsampled * 0.2))
                target_train_size = num_subsampled - target_test_size

                if target_train_size < 1:
                    # Edge case fallback: if math resulted in 0 train samples,
                    # put the remaining data strictly into training.
                    client_train_partitions.append(subsampled_hf)
                    continue

                # Split using absolute integer sizes
                train_test = subsampled_hf.train_test_split(
                    test_size=target_test_size,
                    seed=seed
                )

                client_train_partitions.append(train_test['train'])
                client_test_partitions.append(train_test['test'])

            # Merge
            client_train_hf = concatenate_datasets(client_train_partitions)
            client_test_hf = concatenate_datasets(client_test_partitions)

            # Validation Split
            if validation_split > 0:
                train_val = client_train_hf.train_test_split(test_size=validation_split, seed=seed)
                client_train_hf = train_val['train']
                client_val_hf = train_val['test']
            else:
                client_val_hf = None

            # --- CACHING STEP ---
            client_dir = os.path.join(cache_dir, f"client_{client_id}")
            client_train_hf.save_to_disk(os.path.join(client_dir, "train"))
            client_test_hf.save_to_disk(os.path.join(client_dir, "test"))
            if client_val_hf:
                client_val_hf.save_to_disk(os.path.join(client_dir, "val"))

            # Convert HF to PyTorch Datasets
            client_train_ds = HFToTupleDataset(client_train_hf, transform=train_transform)
            client_test_ds = HFToTupleDataset(client_test_hf, transform=test_transform)

            # Save testing loaders
            client_test_dataloaders[client_id] = DataLoader(client_test_ds, batch_size=total_batch_size, shuffle=False)
            if client_val_hf:
                client_val_ds = HFToTupleDataset(client_val_hf, transform=test_transform)
                client_val_dataloaders[client_id] = DataLoader(client_val_ds, batch_size=total_batch_size,
                                                               shuffle=False)

            # Build partition map
            train_datasets_list.append(client_train_ds)
            num_items = len(client_train_ds)
            partition_map[client_id] = list(range(current_offset, current_offset + num_items))
            current_offset += num_items

            print(f"Client {client_id} processed & cached: {num_items} training samples.")

    # Construct the global loader
    combined_train_ds = ConcatDataset(train_datasets_list)
    client_aware_ds = ClientAwareDataset(combined_train_ds, partition_map)
    fed_sampler = FederatedBatchSampler(partition_map, mini_batch_size, shuffle=True)

    global_train_dataloader = DataLoader(
        client_aware_ds,
        batch_sampler=fed_sampler,
        num_workers=global_args['num_workers_finetuning'],
        pin_memory=True,
        collate_fn=federated_collate_fn
    )

    base_model = get_base_model(global_args, device=device)
    base_model.to(device)

    compression_method = global_args.get('compression_method', 'ae')

    if compression_method == 'ae':
        is_train_ae_on_downstream_data = global_args['dataset'] == global_args['ae_pretrain_dataset']

        # We only reuse the full dataset if AE pretraining uses the same dataset
        auto_encoder_model, ae_pretrain_results = get_auto_encoder(global_args, base_model, device,
                                                                   None)
        param_count_total = auto_encoder_model.get_parameter_count()

        auto_encoder_model.freeze(freeze_encoder=global_args['ae_freeze_encoder_during_finetuning'],
                                  freeze_decoder=global_args['ae_freeze_decoder_during_finetuning'])
        param_count_trainable = auto_encoder_model.get_parameter_count()

        ae_pretrain_results["param_count_trainable"] = param_count_trainable
        ae_pretrain_results["param_count_total"] = param_count_total
        print(f"AE consists of {param_count_trainable} / {param_count_total} trainable parameters")

        if global_args.get('ae_pretrain_only', False):
            print("Skipping fine-tuning as ae_pretrain_only is set to true")
            return
    else:
        print("Using Heuristic Compression. Skipping AE pre-training.")
        auto_encoder_model = None
        ae_pretrain_results = {"param_count_trainable": 0, "param_count_total": 0}



    (client_model, server_model, client_model_requires_any_grad), trainer = models.get_split_model_pair_and_trainer(
        global_args, device, global_args['compress_cls_token'], base_model, auto_encoder_model)
    server_model = server_model.switch_to_device(device)

    client_models = dict()
    client_optimizers_main = dict()
    client_schedulers_main = dict()
    client_optimizers_ae = dict()
    client_schedulers_ae = dict()


    max_nr_of_batches_in_epoch = 0

    print(
        f'Number of trainable params server model: {sum(p.numel() for p in server_model.parameters() if p.requires_grad)} | Total number of parameters: {sum(p.numel() for p in server_model.parameters())}')
    print(
        f'Number of trainable params client model: {sum(p.numel() for p in client_model.parameters() if p.requires_grad)} | Total number of parameters: {sum(p.numel() for p in client_model.parameters())}')

    for client_id in range(global_args['nr_of_clients']):
        _client_model = client_model if client_id == 0 else copy.deepcopy(client_model)
        _client_model = _client_model.switch_to_device(device)
        client_optimizer_main, client_scheduler_main, client_optimizer_ae, client_scheduler_ae = get_optimzer_and_scheduler_for_seperate_ae_finetuning(_client_model, global_args)
        client_optimizers_main[client_id] = client_optimizer_main
        client_optimizers_ae[client_id] = client_optimizer_ae
        client_models[client_id] = _client_model

        if client_model_requires_any_grad:
            client_schedulers_main[client_id] = client_scheduler_main
            client_schedulers_ae[client_id] = client_scheduler_ae

        max_nr_of_batches_in_epoch = len(global_train_dataloader)

    server_optimizer, server_scheduler = get_optimizer_and_scheduler(server_model, global_args)

    start_time = time.time()


    # Run the fine-tuning Loop
    finetune_results = finetune_mpsl(
        global_args=global_args,
        device=device,
        server_model=server_model,
        client_models=client_models,
        trainer=trainer,
        global_train_dataloader=global_train_dataloader,
        val_dataloaders=client_val_dataloaders,
        test_dataloaders=client_test_dataloaders,
        client_optimizers_main=client_optimizers_main,
        client_schedulers_main=client_schedulers_main,
        client_optimizers_ae=client_optimizers_ae,
        client_schedulers_ae=client_schedulers_ae,
        server_optimizer=server_optimizer,
        server_scheduler=server_scheduler,
        client_model_requires_any_grad=client_model_requires_any_grad,
        max_nr_of_batches_in_epoch=max_nr_of_batches_in_epoch,
        search_space_args=search_space_args,
        validation_mode=validation_mode
    )

    # Save Combined Results
    save_combined_experiment_results(finetune_results, ae_pretrain_results, search_space_args,
                                     global_args['save_file_name'])


    end_time = time.time()
    print(f"Fine-tuning completed in: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    global_args = setup_arguments()
    run_2_stage_mpsl_personalized(global_args, None)