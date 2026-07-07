import json
import os

import torch

from ae_trainers.implementations.ae_experiment_results import ExperimentResultsAE
from trainers.implementations.experiment_results import ExperimentResults
from utils.file_utils import save_object_to_json
from utils.fl_utils import AGGREGATED_MODEL_NAME
from utils.mpsl_utils import get_client_name
from pathlib import Path

def _save_object(obj, file_name):
    if file_name is None or len(file_name) == 0:
        raise Exception('file_name should not be None or an empty string.')

    file_name = f'{file_name}.pth'

    torch.save(obj, os.path.join(os.environ['MODEL_WEIGHTS_DIR'], file_name))
    # print(f'Saved model in filename: {file_name}')

def _delete_object(file_name):
    if file_name is None or len(file_name) == 0:
        raise Exception('file_name should not be None or an empty string.')

    file_name = f'{file_name}.pth'
    file_path = Path(file_name)

    # Check if it exists before deleting to avoid errors
    if file_path.exists():
        file_path.unlink()
        # print("File deleted successfully.")
    else:
        print(f"The file {file_name} does not exist.")


def _load_checkpoint(file_name, device):
    if file_name is None or len(file_name) == 0:
        raise Exception('file_name should not be None or an empty string.')

    file_name = f'{file_name}.pth'
    checkpoint = torch.load(os.path.join(os.environ['MODEL_WEIGHTS_DIR'], file_name), map_location=device)

    return checkpoint


def save_centralized_model(centralized_model, desired_file_name):
    _save_object(centralized_model.state_dict(), desired_file_name)

def save_ae_model(autoencoder_model, desired_file_name):
    _save_object(autoencoder_model.state_dict(), desired_file_name)

def delete_model(desired_file_name):
    _delete_object(os.path.join(os.environ['MODEL_WEIGHTS_DIR'], desired_file_name))

def save_split_model(client_model, server_model, desired_file_name):
    _save_object(
        {
            'client_model': client_model.state_dict(),
            'server_model': server_model.state_dict()
        },
        desired_file_name
    )


def save_federated_model(aggregated_model, client_models: dict, desired_file_name):
    save_object = {
            AGGREGATED_MODEL_NAME: aggregated_model.state_dict(),
    }

    for client_id in client_models.keys():
        save_object[get_client_name(client_id)] = client_models[client_id].state_dict()

    _save_object(
        save_object,
        desired_file_name
    )


def load_centralized_model(centralized_model, device, checkpoint_file_name):
    checkpoint = _load_checkpoint(checkpoint_file_name, device)

    centralized_model.load_state_dict(checkpoint)

    return centralized_model


def load_split_model(client_model, server_model, device, checkpoint_file_name):
    checkpoint = _load_checkpoint(checkpoint_file_name, device)

    if client_model is not None:
        client_model.load_state_dict(checkpoint['client_model'])
    server_model.load_state_dict(checkpoint['server_model'])

    return client_model, server_model


def load_federated_model(aggregated_model, client_models: dict, device, checkpoint_file_name):
    """
    :param client_models: A dict consisting of instances of the client models, into which the checkpoint will be loaded.
    :param aggregated_model: An instance of the aggregated_model model, into which the checkpoint will be loaded.
    :return: a tuple of the form (aggregated_model, dict<client_models>) initialized with the desired weights.
    """
    checkpoint = _load_checkpoint(checkpoint_file_name, device)

    aggregated_model.load_state_dict(checkpoint[AGGREGATED_MODEL_NAME])

    for idx in range(len(client_models)):
        if idx in client_models:
            client_models[idx].load_state_dict(checkpoint[get_client_name(idx)])

    return aggregated_model, client_models


def save_experiment_results(experiment_results: ExperimentResults, desired_file_name):
    """
    Attempts to save the results object that is a json object of all results (to provide more ease of use).
    """
    if desired_file_name is None:
        raise Exception('desired_file_name was None: cannot save results dict')
    else:
        full_file_path = os.path.join(os.environ["MODEL_WEIGHTS_DIR"], desired_file_name + '.json')
        nested_data = {
            "finetune_results": experiment_results.to_json(),
        }
        save_object_to_json(nested_data, full_file_path)


def save_experiment_results_with_args(experiment_results: ExperimentResults, search_space_args, desired_file_name):
    """
    Attempts to save the results object that is a json object of all results (to provide more ease of use).
    """
    if desired_file_name is None:
        raise Exception('desired_file_name was None: cannot save results dict')
    else:
        full_file_path = os.path.join(os.environ["MODEL_WEIGHTS_DIR"], desired_file_name + '.json')
        nested_data = {
            "finetune_results": experiment_results.to_json(),
        }
        save_object_to_json(nested_data, full_file_path)

def load_experiment_results(file_name, validation_mode):
    """
    Loads the experiment results from a JSON file and returns it as a dictionary.
    """
    if file_name is None:
        raise Exception('file_name was None: cannot load results dict')
    else:
        full_file_path = os.path.join(os.environ["MODEL_WEIGHTS_DIR"], file_name + '.json')
        with open(full_file_path, 'r') as f:
            data = json.load(f)

            experiment_results = ExperimentResults(validation_mode=validation_mode)
            experiment_results.from_json(data['finetune_results'])

            return experiment_results



def save_combined_experiment_results(experiment_results, ae_results, search_space_args, desired_file_name):
    """
    Combines experiment results and ae_results into one JSON file.
    """
    if desired_file_name is None:
        raise Exception('desired_file_name was None: cannot save results dict')

    # 1. Construct the path
    full_file_path = os.path.join(os.environ["MODEL_WEIGHTS_DIR"], f"{desired_file_name}.json")

    # 2. Combine the data into a single dictionary
    combined_data = {
        "params": search_space_args,
        "finetune_results": experiment_results.to_json(),
        "ae_pretrain_results": ae_results
    }

    # 3. Save the combined object
    save_object_to_json(combined_data, full_file_path)

def save_ae_experiment_results(experiment_results: ExperimentResultsAE, filename):
    """
    Attempts to save the results object that is a json object of all results (to provide more ease of use).
    """
    if filename is None:
        raise Exception('desired_file_name was None: cannot save results dict')
    else:
        full_file_path = os.path.join(os.environ['AE_WEIGHTS_DIR'], filename, 'results.json')

        save_object_to_json(experiment_results.to_json(), full_file_path)