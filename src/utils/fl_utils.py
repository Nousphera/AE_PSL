import copy
import torch

NR_OF_BYTES_PER_FLOAT32 = 4
AGGREGATED_MODEL_NAME = 'aggregated_model'


def fed_avg(models: dict, client_weight_multipliers: dict, aggregated_model=None):
    # Taking any arbitrary client_model as the base will do, as all parameters will be modified anyways. We'll resort to simply choosing the first model.
    aggregated_model = copy.deepcopy(models[0]) if aggregated_model is None else copy.deepcopy(aggregated_model)

    nr_of_clients = len(models.keys())

    for key in aggregated_model.state_dict().keys():
        aggregated_model_parameter = aggregated_model.state_dict(keep_vars=True)[key]

        temp = torch.zeros_like(aggregated_model_parameter, dtype=torch.float32)

        for client_id in range(nr_of_clients):
            temp += client_weight_multipliers[client_id] * models[client_id].state_dict()[key]

        aggregated_model_parameter.data.copy_(temp)

    return aggregated_model


def get_client_weight_multipliers__nr_of_elements(nr_of_elements_per_client: dict):
    """
    Returns a dictionary of weight multipliers for each client for use in the FedAvg algorithm.
    Each multiplier is the proportion of the client's data relative to the total number of data points across all clients.
    """
    nr_of_total_elements = sum(nr_of_elements_per_client.values())

    return {client_id: nr_of_elements_per_client[client_id] / nr_of_total_elements for client_id in nr_of_elements_per_client.keys()}


def synchronize_models_in_place(aggregated_model, client_models: dict):
    for key in aggregated_model.state_dict().keys():
        for client_model in client_models.values():
            client_model.state_dict()[key].data.copy_(aggregated_model.state_dict()[key])


def get_communication_size_for_model_in_bytes(model, only_count_trainable_parameters=False):
    """Represents the communication size of communicating the given model's parameters. Note that this is not equal to the memory size of loading the model in-memory due to overhead, especially not when using CUDA."""
    total_relevant_params = sum(p.numel() for p in model.parameters() if (p.requires_grad if only_count_trainable_parameters else True))

    return total_relevant_params * NR_OF_BYTES_PER_FLOAT32
