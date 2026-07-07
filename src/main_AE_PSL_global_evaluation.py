import copy
import time
from argparse import Namespace

import torch

import available_datasets as datasets
import models
from ae_trainers.ae_trainer import get_auto_encoder
from models import get_base_model
from trainers.implementations.experiment_results import ExperimentResults
from utils.orchestrator_argument_utils import build_base_argument_parser, \
    expand_argument_parser_with_distributed_learning_parameters, set_env_variables, \
    namespace_to_dict
from utils.config_utils import set_random_seed
from utils.cuda_utils import get_device
from utils.dataloader_utils import get_distributed_dataloaders_from_datasets
from utils.fl_utils import fed_avg, get_client_weight_multipliers__nr_of_elements
from utils.model_saving_utils import save_split_model, save_experiment_results, \
    load_split_model, delete_model, save_combined_experiment_results
from utils.mpsl_utils import compute_mini_batch_size
from utils.scheduler_utils import get_optimizer_and_scheduler, get_optimzer_and_scheduler_for_seperate_ae_finetuning





def setup_arguments() -> dict:
    parser = build_base_argument_parser()
    parser = expand_argument_parser_with_distributed_learning_parameters(parser)

    parser.add_argument('--test_num_workers', type=int, default=5,
                        help='num_workers provided to the test Dataloader. For Split Learning, we differentiate between num_workers for the train Dataloader and test_num_workers for the test Dataloader.')

    args: Namespace = parser.parse_args()
    set_env_variables(args)
    global_args = namespace_to_dict(args)
    return global_args



def finetune_mpsl(global_args, device, server_model, client_models, trainer,
                  global_train_dataloader, val_dataloader, test_dataloader,
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

        trainer.warmup_epoch(
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

            trainer.client_specific_alignment(
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


        # return experiment_results

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

        # --- Aggregation ---
        aggregated_client_model = fed_avg(client_models,
                                          get_client_weight_multipliers__nr_of_elements(nr_of_elements_per_client_dict))

        # --- Regular Checkpointing ---
        if global_args['save_model_after_each_epoch']:
            save_split_model(aggregated_client_model, server_model, global_args['save_file_name'])

        # --- Validation & Testing ---
        with torch.no_grad():

            # Option 1: Early Saving / Stopping using Validation Set
            if val_dataloader is not None and validation_mode in ['early_saving', 'early_stopping']:

                val_loss, val_acc = trainer.test_epoch_global_evaluation(aggregated_client_model, server_model, val_dataloader, device, experiment_results, epoch_nr, use_gpu_transform, global_args['use_compression_during_global_eval'])


                print(f"Epoch {epoch_nr} --- Validation --- Loss: {val_loss:.3f} - Acc: {val_acc * 100.0:.2f}%")

                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # print(f"Validation loss improved to {best_val_loss:.6f}. Saving model...")
                    save_split_model(aggregated_client_model, server_model, global_args['save_file_name'])
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
                test_loss, test_acc = trainer.test_epoch_global_evaluation(aggregated_client_model, server_model, test_dataloader, device,
                                                       experiment_results, epoch_nr,
                                                       use_gpu_transform, global_args['use_compression_during_global_eval'])

                # IMPORTANT: Update final stats during the loop for non-validation mode
                experiment_results.final_test_loss = test_loss
                experiment_results.final_test_accuracy = test_acc

                print(f"Epoch {epoch_nr} --- Test --- Loss: {test_loss:.3f} - Acc: {test_acc * 100.0:.2f}%")

        save_experiment_results(experiment_results, global_args['save_file_name'])

    print(f'Finished training phase.')

    # --- Final Test Evaluation ---
    # If early saving/stopping was used, reload the best model
    if validation_mode in ['early_saving', 'early_stopping']:
        print("Loading best model for final testing...")
        try:
            # Load weights into the aggregated model and server model
            load_split_model(aggregated_client_model, server_model, device, global_args['save_file_name'])
        except FileNotFoundError:
            print("Warning: Best model file not found. Using current weights.")

    print('Starting Final Test Set Evaluation')
    with torch.no_grad():
        final_test_loss, final_test_acc = trainer.test_epoch_global_evaluation(aggregated_client_model, server_model, test_dataloader, device,
                                                 None, global_args['nr_of_epochs'],
                                                 use_gpu_transform, global_args['use_compression_during_global_eval'])
        # experiment_results.final_test_loss = final_test_loss
        experiment_results.final_test_accuracy = final_test_acc
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


def run_2_stage_mpsl(global_args: dict, search_space_args: dict):
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

    full_dataset = datasets.load_data(name=global_args['dataset'], num_partitions=global_args['nr_of_clients'],
                                      split=global_args['dataset_split_type'], seed=global_args['random_seed'],
                                      global_args=global_args, val_split=validation_split)

    val_dataset = full_dataset.load_validation_set()
    test_dataset = full_dataset.load_test_set()

    base_model = get_base_model(global_args, device=device)
    base_model.to(device)

    compression_method = global_args.get('compression_method', 'ae')



    if compression_method == 'ae':
        is_train_ae_on_downstream_data = global_args['dataset'] == global_args['ae_pretrain_dataset']

        # We only reuse the full dataset if AE pretraining uses the same dataset
        auto_encoder_model, ae_pretrain_results = get_auto_encoder(global_args, base_model, device,
                                                                   full_dataset if is_train_ae_on_downstream_data else None)
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

    global_train_dataloader, val_dataloader, test_dataloader = get_distributed_dataloaders_from_datasets(
        train_dataset=full_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        validation_mode=validation_mode,
        mini_batch_size=mini_batch_size,
        total_batch_size=total_batch_size,
        num_workers=global_args['num_workers_finetuning'],
        dataloader_class=datasets.DataLoader,
        nr_of_clients=global_args['nr_of_clients'],
        small_test_run=global_args['small_test_run'],
        random_seed=global_args['random_seed'],
        collate_fn=full_dataset.get_collate_fn()
    )

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
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
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
    run_2_stage_mpsl(global_args, None)