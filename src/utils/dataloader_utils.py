
import torch
from torch.utils.data import random_split

import available_datasets




# Handles the validation set logic
# For AEs, we don't use the distributedDataset class, hence we do the splitting manually here
def get_dataloaders_from_datasets_AE(train_dataset, test_dataset, validation_mode, validation_split, batch_size, num_workers, dataloader_class, collate_fn=None):
    val_dataloader = None

    # Using a validation set, so create a validation dataloader
    if validation_mode != 'none':
        # move to utils later
        train_size = int((1.0 - validation_split) * len(train_dataset))
        train_dataset, val_dataset = random_split(train_dataset,
                                                            [train_size, len(train_dataset) - train_size])
        val_dataloader = dataloader_class(val_dataset, batch_size=batch_size,
                                         shuffle=False, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)

    # Don't need to use the custom distributed dataset class here, as we are using activations only.
    train_dataloader = dataloader_class(train_dataset, batch_size=batch_size,
                                       shuffle=True, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader = dataloader_class(test_dataset, batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    return train_dataloader, val_dataloader, test_dataloader

# Since we use distributedDataset, we should pass the predetermined validation dataset here
def get_dataloaders_from_datasets(train_dataset, validation_dataset, test_dataset, validation_mode, batch_size, num_workers, dataloader_class, small_test_run, collate_fn=None):

    if small_test_run:
        train_dataset = available_datasets.Subset(train_dataset, range(0, len(train_dataset) // 20))
        validation_dataset = available_datasets.Subset(validation_dataset, range(0, len(validation_dataset) // 20))
        test_dataset = available_datasets.Subset(test_dataset, range(0, len(test_dataset) // 20))

    # Using a validation set, so create a validation dataloader
    if validation_mode != 'none':
        val_dataloader = dataloader_class(validation_dataset, batch_size=batch_size,
                                         shuffle=False, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    else:
        val_dataloader = None

    # Don't need to use the custom distributed dataset class here, as we are using activations only.
    train_dataloader = dataloader_class(train_dataset, batch_size=batch_size,
                                       shuffle=True, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader = dataloader_class(test_dataset, batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
    # print(len(train_dataloader), len(val_dataloader), len(test_dataloader))
    return train_dataloader, val_dataloader, test_dataloader


import torch
from torch.utils.data import Sampler
from collections import defaultdict


class ClientAwareDataset(torch.utils.data.Dataset):
    """Wraps the base dataset to also return the client_id."""

    def __init__(self, base_dataset, partition_id_to_indices):
        self.base_dataset = base_dataset
        # Create a reverse lookup: index -> client_id
        self.index_to_client = {}
        for client_id, indices in partition_id_to_indices.items():
            for idx in indices:
                self.index_to_client[idx] = client_id

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        X, y = self.base_dataset[idx]
        client_id = self.index_to_client[idx]
        return X, y, client_id


class FederatedBatchSampler(Sampler):
    def __init__(self, partition_id_to_indices, mini_batch_size, shuffle):
        self.partition_id_to_indices = partition_id_to_indices
        self.mini_batch_size = mini_batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Create a copy of indices so we can pop from them
        client_indices = {}
        for client_id, indices in self.partition_id_to_indices.items():
            idx_copy = list(indices)
            if self.shuffle:
                # torch.randperm(len(idx_copy)).tolist()  # Optional: shuffle
                generator = torch.Generator().manual_seed(42 + client_id)
                perm = torch.randperm(len(idx_copy), generator=generator).tolist()
                idx_copy = [idx_copy[i] for i in perm]
            client_indices[client_id] = idx_copy

        while True:
            batch_indices = []
            active_clients = 0

            # Draw up to mini_batch_size from every client still having data
            for client_id, indices in client_indices.items():
                if len(indices) > 0:
                    active_clients += 1
                    # Take up to mini_batch_size
                    take_count = min(self.mini_batch_size, len(indices))
                    batch_indices.extend(indices[:take_count])
                    # Remove used indices
                    client_indices[client_id] = indices[take_count:]

            if active_clients == 0:
                break  # All clients are out of data

            yield batch_indices

    def __len__(self):
        # Estimate total batches (based on the largest client partition)
        max_len = max(len(v) for v in self.partition_id_to_indices.values())
        return (max_len + self.mini_batch_size - 1) // self.mini_batch_size


def federated_collate_fn(batch):
    """Groups the batch by client_id."""
    # batch is a list of tuples: (X, y, client_id)
    client_batches = defaultdict(list)
    for X, y, client_id in batch:
        client_batches[client_id].append((X, y))

    result = {}
    for client_id, items in client_batches.items():
        # Stack X and y for this specific client

        X_stacked = torch.stack([item[0] for item in items])
        # Note: if y is an int, use torch.tensor. If it's already a tensor, use torch.stack
        y_stacked = torch.tensor([item[1] for item in items])
        result[client_id] = (X_stacked, y_stacked)

    return result



# each client gets it's own client dataloader, with a partition of the train dataset
def get_distributed_dataloaders_from_datasets(train_dataset, validation_dataset, test_dataset, validation_mode, mini_batch_size, total_batch_size, num_workers, dataloader_class, nr_of_clients, small_test_run, random_seed, collate_fn=None):

    val_dataloader = None

    client_dataloaders = dict()

    # if small_test_run:
    #     train_dataset = available_datasets.Subset(train_dataset, range(0, len(train_dataset) // 50))
    #     validation_dataset = available_datasets.Subset(validation_dataset, range(0, len(validation_dataset) // 50)) if validation_dataset else None
    #     test_dataset = available_datasets.Subset(test_dataset, range(0, len(test_dataset) // 50))

    # Force the partitioner to calculate indices
    train_dataset.partitioner._determine_partition_id_to_indices_if_needed()
    partition_map = train_dataset.partitioner._partition_id_to_indices

    # Wrap dataset and create the global Sampler
    client_aware_ds = ClientAwareDataset(train_dataset.train_ds, partition_map)
    fed_sampler = FederatedBatchSampler(partition_map, mini_batch_size, shuffle=False)

    # A SINGLE Dataloader for all clients
    global_train_dataloader = dataloader_class(
        client_aware_ds,
        batch_sampler=fed_sampler,  # Replaces batch_size and shuffle
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=federated_collate_fn,
        generator=torch.Generator().manual_seed(random_seed),
    )


    if validation_mode != 'none':
        val_dataloader = dataloader_class(validation_dataset, batch_size=total_batch_size,
                                      shuffle=False, pin_memory=True, num_workers=num_workers,
                                      collate_fn=collate_fn, generator=torch.Generator().manual_seed(random_seed))


    test_dataloader = dataloader_class(test_dataset, batch_size=total_batch_size,
                                       shuffle=False,
                                       pin_memory=True, num_workers=num_workers, collate_fn=collate_fn, generator=torch.Generator().manual_seed(random_seed))

    # plot_client_distribution(all_distributions, num_classes=101)

    return global_train_dataloader, val_dataloader, test_dataloader


def get_distributed_dataloaders_from_datasets_personalized(train_dataset, validation_dataset, test_dataset, validation_mode, mini_batch_size, total_batch_size, num_workers, dataloader_class, nr_of_clients, small_test_run, random_seed, collate_fn=None):

    val_dataloader = None

    client_dataloaders = dict()

    if small_test_run:
        # train_dataset = available_datasets.Subset(train_dataset, range(0, len(train_dataset) // 50))
        validation_dataset = available_datasets.Subset(validation_dataset, range(0, len(validation_dataset) // 50))
        test_dataset = available_datasets.Subset(test_dataset, range(0, len(test_dataset) // 50))

    # Force the partitioner to calculate indices
    train_dataset.partitioner._determine_partition_id_to_indices_if_needed()
    partition_map = train_dataset.partitioner._partition_id_to_indices

    # Wrap dataset and create the global Sampler
    client_aware_ds = ClientAwareDataset(train_dataset.train_ds, partition_map)
    fed_sampler = FederatedBatchSampler(partition_map, mini_batch_size)

    # A SINGLE Dataloader for all clients
    global_train_dataloader = dataloader_class(
        client_aware_ds,
        batch_sampler=fed_sampler,  # Replaces batch_size and shuffle
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=federated_collate_fn,
        generator=torch.Generator().manual_seed(random_seed),
    )


    if validation_mode != 'none':
        val_dataloader = dataloader_class(validation_dataset, batch_size=total_batch_size,
                                      shuffle=False, pin_memory=True, num_workers=num_workers,
                                      collate_fn=collate_fn, generator=torch.Generator().manual_seed(random_seed))


    test_dataloader = dataloader_class(test_dataset, batch_size=total_batch_size,
                                       shuffle=False,
                                       pin_memory=True, num_workers=num_workers, collate_fn=collate_fn, generator=torch.Generator().manual_seed(random_seed))

    # plot_client_distribution(all_distributions, num_classes=101)

    return global_train_dataloader, val_dataloader, test_dataloader


class HFToTupleDataset(torch.utils.data.Dataset):
    """Converts a HuggingFace dataset into PyTorch (X, y) tuples."""

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        label = item['character']  # FEMNIST target column

        if self.transform:
            image = self.transform(image)

        return image, label