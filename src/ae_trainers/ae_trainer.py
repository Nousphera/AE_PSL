import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange

import available_datasets
from ae_trainers.implementations.ae_experiment_results import ExperimentResultsAE
from ae_trainers.implementations.classification.centralized_ae_trainer import CentralizedAETrainer
from available_datasets import DistributedDataset
from models.auto_encoder import initialize_AE, BaseViTAE
from utils.ae_registry_utils import load_auto_encoder_model, prepare_ae_dir, filename_from_signature, \
    save_ae_with_signature
from utils.dataloader_utils import get_dataloaders_from_datasets_AE
from utils.model_saving_utils import save_ae_experiment_results
from utils.scheduler_utils import get_ae_pretrain_optimizer_and_scheduler


activations = {}

# Makes the choice to pretrain or load an AE model based on global_args
def get_auto_encoder(global_args, base_model, device, full_dataset : DistributedDataset = None):

    # This signature is used to identify the AE model. It will be used to find an existing AE model or to save a new one.
    ae_signature = {
        'type': global_args['ae_type'],
        'dataset': global_args['ae_pretrain_dataset'],
        'dataset_proportion': global_args['ae_pretrain_dataset_fraction'],
        'model': global_args['model'],
        'split_layer': global_args['split_layer'],
        'input_dim': base_model.get_hidden_dim(),
        'latent_dim': global_args['ae_latent_dim'],
    }

    # Initialize an untrained AE model
    # if latent dim == hidden dim of model, then it will initialize as an identity bottleneck regardless of model type
    auto_encoder_model = initialize_AE(global_args, base_model.get_hidden_dim())
    auto_encoder_model.to(device)

    if not global_args['ae_general_alignment']:
        return auto_encoder_model, {}

    if type(auto_encoder_model) is BaseViTAE:
        print("AE type is BaseViTAE, skipping AE pretraining/loading.")
        return auto_encoder_model, {}


    if not global_args['ae_force_retrain']:
        # Load an existing AE model based on required ae signature
        model, results = load_auto_encoder_model(global_args, auto_encoder_model, ae_signature, device)
        if model is not None:
            return model, results
        else:
            print("No existing AE model found, proceeding to pretrain a new model.")
    else:
        print("AE force retrain flag is set, proceeding to pretrain a new model.")


    # If the AE pretrain dataset differs, then we are training on downstream client data
    is_train_ae_on_downstream_data = global_args['dataset'] == global_args['ae_pretrain_dataset']

    if is_train_ae_on_downstream_data:
        print(f"Note: AE pre-training on downstream client dataset: {global_args['dataset']}")
    else:
        print(f"Note: AE pre-training on separate pre-train dataset: {global_args['ae_pretrain_dataset']}")


    full_dataset = available_datasets.load_data(name=global_args['ae_pretrain_dataset'], num_partitions=1,
                                                split='iid', seed=global_args['random_seed'],
                                                global_args=global_args)


    train_dataset = full_dataset.load_partition(partition_id=0)
    test_dataset = full_dataset.load_test_set()

    if global_args['small_test_run']:
        train_dataset = available_datasets.Subset(train_dataset, range(0, len(train_dataset) // 100))
        test_dataset = available_datasets.Subset(test_dataset, range(0, len(test_dataset) // 100))
    else:
        # Take the correct proportion of the dataset, as we want to experiment with how much data is needed to pretrain the AE
        train_dataset = available_datasets.Subset(train_dataset, range(0, int(len(train_dataset) * global_args[
            'ae_pretrain_dataset_fraction'])))
        test_dataset = available_datasets.Subset(test_dataset, range(0, int(len(test_dataset) * global_args[
            'ae_pretrain_dataset_fraction'])))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=global_args['ae_pretrain_batch_size'], shuffle=True,
                                  pin_memory=True,
                                  num_workers=global_args['num_workers_activation_loading'],
                                  collate_fn=full_dataset.get_collate_fn())

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=global_args['ae_pretrain_batch_size'], shuffle=False,
                                 pin_memory=True,
                                 num_workers=global_args['num_workers_activation_loading'],
                                 collate_fn=full_dataset.get_collate_fn())

    start_time = time.time()


    # Ensure activation cache directory exists and try loading cached activations first.
    activations_dir = os.path.join('data', 'activations')
    os.makedirs(activations_dir, exist_ok=True)

    base_name = f"acts_ds={ae_signature['dataset']}_model={ae_signature['model']}_split={ae_signature['split_layer']}_frac={ae_signature['dataset_proportion']}_small={global_args['small_test_run']}"
    train_acts_path = os.path.join(activations_dir, f"{base_name}_train.pt")
    test_acts_path = os.path.join(activations_dir, f"{base_name}_test.pt")

    if os.path.exists(train_acts_path) and os.path.exists(test_acts_path):
        print(
            f"Loading cached activations from `data/activations`: {os.path.basename(train_acts_path)}, {os.path.basename(test_acts_path)}")
        train_tensor = torch.load(train_acts_path)
        test_tensor = torch.load(test_acts_path)
        train_acts_dataset = TensorDataset(train_tensor)
        test_acts_dataset = TensorDataset(test_tensor)
    else:
        print("Cached activations not found. Extracting activations and saving to `data/activations`.")
        train_acts_dataset = extract_activations(base_model, global_args['split_layer'], train_dataloader, device)
        test_acts_dataset = extract_activations(base_model, global_args['split_layer'], test_dataloader, device)

        # Save the underlying tensors (extracted activations are on CPU already)
        try:
            torch.save(train_acts_dataset.tensors[0], train_acts_path)
            torch.save(test_acts_dataset.tensors[0], test_acts_path)
            print(
                f"Saved activations to `data/activations` as {os.path.basename(train_acts_path)}, {os.path.basename(test_acts_path)}")
        except Exception as e:
            print(f"Warning: failed to save activations cache: {e}")



    # Calculate duration
    end_time = time.time()
    duration = end_time - start_time

    print(f"Activation extraction completed in: {duration:.2f} seconds")

    return pretrain_auto_encoder(global_args, base_model, auto_encoder_model, train_acts_dataset, test_acts_dataset, ae_signature, device)




def pretrain_auto_encoder(global_args, base_model, auto_encoder_model, train_acts_dataset, test_acts_dataset,
                          ae_signature, device,
                          early_stopping_patience=3):
    """
    Args:
        validation_mode (str): Options are 'none', 'early_saving', 'early_stopping'.
        model_save_path (str): Path to save the best model checkpoint.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
    """
    # ga validation_mode
    # val_split

    validation_mode = global_args['val_mode']
    validation_split = global_args['val_split']
    if validation_split == 0: validation_mode = 'none'

    train_acts_dataloader, val_acts_dataloader, test_acts_dataloader = get_dataloaders_from_datasets_AE(train_acts_dataset, test_acts_dataset, validation_mode, validation_split,
                                  global_args['ae_pretrain_batch_size'], global_args['num_workers_ae_pretraining'], DataLoader)

    auto_encoder_trainer = CentralizedAETrainer()

    optimizer, scheduler = get_ae_pretrain_optimizer_and_scheduler(auto_encoder_model, global_args)
    loss_fn = torch.nn.MSELoss()

    ae_experiment_results = ExperimentResultsAE(validation_mode)
    model_save_dir = prepare_ae_dir(ae_signature)
    model_save_path = os.path.join(model_save_dir, 'ae_model')

    # Validation State Variables
    best_val_loss = float('inf')
    patience_counter = 0

    print("\nStarting AE Pre-training\n")

    start_time = time.time()

    # # # # # # # # # # # # # # # # # Training # # # # # # # # # # # # # # # # #
    with trange(global_args['ae_pretrain_epochs'], unit='epoch') as epochs:
        for epoch_nr in epochs:

            train_loss = auto_encoder_trainer.train_epoch(
                experiment_results=ae_experiment_results,
                auto_encoder=auto_encoder_model,
                base_model=base_model,
                split_layer=global_args['split_layer'],
                device=device,
                dataloader=train_acts_dataloader,
                epoch_nr=epoch_nr,
                optimizer=optimizer,
                loss_fn=loss_fn
            )
            # print(f"Epoch {epoch_nr} --- Train --- Reconstruction Loss: {train_loss:.3f}")

            scheduler.step()

            if val_acts_dataloader is not None and validation_mode in ['early_saving', 'early_stopping']:
                with torch.no_grad():
                    val_loss = auto_encoder_trainer.test_epoch(
                        experiment_results=ae_experiment_results,
                        auto_encoder=auto_encoder_model,
                        base_model=base_model,
                        split_layer=global_args['split_layer'],
                        device=device,
                        dataloader=val_acts_dataloader,
                        epoch_nr=epoch_nr,
                        loss_fn=loss_fn
                    )
                    # print(f"Epoch {epoch_nr} --- Validation --- Reconstruction Loss: {train_loss:.3f}")

                    # Check improvement
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # print(f"Validation loss improved to {best_val_loss:.6f}. Saving model to {model_save_path}...")
                        if model_save_path:
                            torch.save(auto_encoder_model.state_dict(), model_save_path)
                    else:
                        patience_counter += 1

                    if validation_mode == 'early_stopping' and patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch_nr} epochs.")
                        break

                    epochs.set_postfix({
                        'Train Loss': f'{train_loss:.4f}',
                        'Val Loss': f'{val_loss:.4f}',
                        'LR': f'{scheduler.get_last_lr()[0]:.1e}',
                    })

            # Option 1: No Val (or just logging test set per epoch like original code)
            elif validation_mode == 'none':
                with torch.no_grad():

                    test_loss = auto_encoder_trainer.test_epoch(
                        experiment_results=ae_experiment_results,
                        auto_encoder=auto_encoder_model,
                        base_model=base_model,
                        split_layer=global_args['split_layer'],
                        device=device,
                        dataloader=test_acts_dataloader,
                        epoch_nr=epoch_nr,
                        loss_fn=loss_fn
                    )
                    ae_experiment_results.final_test_metric = test_loss
                    # print(f"Epoch {epoch_nr} --- Test --- Loss: {test_loss:.3f}\n")

                    epochs.set_postfix({
                        'Train Loss': f'{train_loss:.4f}',
                        'Test Loss': f'{test_loss:.4f}',
                        'LR': f'{scheduler.get_last_lr()[0]:.1e}',
                    })

            save_ae_experiment_results(ae_experiment_results, filename_from_signature(ae_signature))

    print(f'Finished training AE.')


    if validation_mode in ['early_saving', 'early_stopping']:
        # print(f"Loading best model from {model_save_path} for final testing...")
        try:
            auto_encoder_model.load_state_dict(torch.load(model_save_path))
        except FileNotFoundError:
            print("Warning: Best model file not found. Using current weights.")

        print('Final AE Test Set Evaluation')
        with torch.no_grad():
            final_test_loss = auto_encoder_trainer.test_epoch(
                experiment_results=None,
                auto_encoder=auto_encoder_model,
                base_model=base_model,
                split_layer=global_args['split_layer'],
                device=device,
                dataloader=test_acts_dataloader,
                epoch_nr=global_args['ae_pretrain_epochs'],
                loss_fn=loss_fn
            )
            ae_experiment_results.final_test_metric = final_test_loss
            print(f"Final Test Metric from saved model --- Loss: {final_test_loss:.3f}")

    if global_args['ae_save_final_weights']:
        save_ae_with_signature(auto_encoder_model, ae_signature)

    save_ae_experiment_results(ae_experiment_results, filename_from_signature(ae_signature))

    # Calculate duration
    end_time = time.time()
    duration = end_time - start_time

    print(f"\nAE pre-training completed in: {duration:.2f} seconds\n")

    ae_experiment_results_json = ae_experiment_results.to_json()
    ae_experiment_results_json['reused_AE'] = False

    return auto_encoder_model, ae_experiment_results_json

def extract_activations(base_model, split_layer, dataloader, device):
    """
    Runs the base_model once over the dataset to extract activations.
    Returns a TensorDataset containing these activations.
    """
    activations_list = []
    base_model.eval()

    print("Pre-computing activations...")
    with torch.no_grad():
        for X, _ in tqdm(dataloader, desc="Extracting"):

            # Get activations
            acts = base_model.retrieve_split_layer_activations(X, split_layer)

            # Move back to CPU to store in RAM (prevents GPU OOM)
            activations_list.append(acts.cpu())

    # Concatenate all batches and wrap in a dataset
    all_activations = torch.cat(activations_list)
    return TensorDataset(all_activations)