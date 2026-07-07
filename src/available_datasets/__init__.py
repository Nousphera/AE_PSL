import typing
import numpy as np
import numpy.typing as npt
import torch
from torchvision import transforms

# Note that this folder is called 'available_datasets' to avoid conflicts with the huggingface 'datasets' package.
from .unimodal.audio_classification.speechcommands import SpeechCommandsV2
from .unimodal.image_classification.cifar100 import CIFAR100
from .unimodal.image_classification.food101 import Food101
from .unimodal.image_classification.imagenet100 import ImageNet100
from .unimodal.image_classification.sun397 import SUN397

import torch.nn.functional as F  # <--- ADD THIS LINE
import torchaudio

dataloaders = {
    # unimodal
    'cifar100': (CIFAR100, 'cifar100'),
    'food101': (Food101, 'food101'),
    'imagenet': (ImageNet100, 'imagenet'),
    'sun397': (SUN397, 'sun397'),
    # 'stanford_cars': (StanfordCars, 'stanford_cars'),
    'femnist': (None, 'femnist'),
    # 'speechcommands': (SpeechCommandsV2, 'speechcommands'),
}


def get_default_vit_transforms(train: bool):
    """
    Returns standard transforms for ViT models.
    Train: Resize -> RandomFlip -> ToTensor -> Normalize
    Val:   Resize -> ToTensor -> Normalize
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            # Optional: transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


def get_vit_mnist_transforms():
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    layers = [
        transforms.Resize((224, 224)),
        # Convert 1 channel to 3 channels
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        normalize,
    ]


    return transforms.Compose(layers)


import torchaudio


def get_ast_transforms(train: bool):
    """
    Transforms audio waveform to normalized Mel-Spectrogram for AST.
    AST's forward method expects (time_frames, frequency_bins).
    """

    class ASTTransform:
        def __init__(self, train):
            self.train = train
            # Standard AST configuration
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=128,
                f_min=50,
                f_max=8000
            )
            self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

        def __call__(self, waveform):
            # 1. Convert to Mel Spectrogram: (channels, n_mels, time)
            mel_spec = self.mel_transform(waveform)
            mel_spec = self.amp_to_db(mel_spec)


            mean = torch.mean(mel_spec)
            std = torch.std(mel_spec)
            mel_spec = (mel_spec - mean) / (std * 2)


            return mel_spec

    return ASTTransform(train)


def get_speechcommands_transforms(train: bool):
    """
    Pads audio to 1s, computes MelSpectrogram (100x128), and normalizes for AST.
    """

    class ASTTransform:
        def __init__(self, train):
            self.train = train
            self.target_length = 16000  # 1 second at 16kHz

            # AST standard config for 16kHz audio
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=400,
                win_length=400,
                hop_length=160,
                f_min=50,
                f_max=8000,
                n_mels=128
            )
            self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

        def __call__(self, waveform):
            # 1. Pad or truncate to exactly 16000 samples
            seq_len = waveform.shape[1]
            if seq_len < self.target_length:
                waveform = F.pad(waveform, (0, self.target_length - seq_len))
            elif seq_len > self.target_length:
                waveform = waveform[:, :self.target_length]

            # 2. Convert to Mel Spectrogram
            mel_spec = self.mel_transform(waveform)
            mel_spec = self.amp_to_db(mel_spec)

            # 3. Standardize (AST expectation)
            mean = torch.mean(mel_spec)
            std = torch.std(mel_spec)
            mel_spec = (mel_spec - mean) / (std * 2)

            # 4. Transpose for AST forward pass: (time_frames, freq_bins) -> (100, 128)
            mel_spec = mel_spec.squeeze(0).transpose(0, 1)

            # Optional: Add torchaudio.transforms.FrequencyMasking / TimeMasking here if self.train

            return mel_spec

    return ASTTransform(train)


def get_train_val_split(dataset, val_split, seed):
    """
    Splits a PyTorch dataset into train and validation subsets using
    torch functions for reproducibility via a torch.Generator.
    """
    if val_split == 0.0:
        return Subset(dataset, torch.arange(len(dataset))), None

    if not (0.0 < val_split < 1.0):
        raise ValueError("val_split must be between 0.0 and 1.0")

    dataset_len = len(dataset)
    val_len = int(dataset_len * val_split)
    train_len = dataset_len - val_len

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(dataset_len, generator=generator)

    train_indices = shuffled_indices[:train_len].tolist()
    val_indices = shuffled_indices[train_len:].tolist()

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset


class Subset(torch.utils.data.Subset):
    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def targets(self):
        return [self.dataset.targets[i] for i in self.indices]

    @property
    def num_rows(self):
        return len(self)


class DataLoader(torch.utils.data.DataLoader):
    @property
    def num_classes(self):
        return self.dataset.num_classes


class DirichletDataPartitioner:
    """
    As per https://github.com/adap/flower
    """

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            num_partitions: int,
            alpha: typing.Union[int, float, typing.List[float], np.ndarray],
            min_partition_size: int = 10,
            self_balancing: bool = False,
            shuffle: bool = True,
            seed: typing.Optional[int] = 42,
            custom_partitioner_function=None
    ) -> None:

        self.dataset = dataset
        self._num_partitions = num_partitions
        self._check_num_partitions_greater_than_zero()
        self._alpha = self._initialize_alpha(alpha)
        self._min_partition_size: int = min_partition_size
        self._self_balancing = self_balancing
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

        self._avg_num_of_samples_per_partition: typing.Optional[float] = None
        self._unique_classes: typing.Optional[typing.Union[typing.List[int], typing.List[str]]] = None
        self._partition_id_to_indices: typing.Dict[int, typing.List[int]] = {}
        self._partition_id_to_indices_determined = False
        self.custom_partitioner_function = custom_partitioner_function

    @property
    def num_partitions(self) -> int:
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    def _check_num_partitions_greater_than_zero(self) -> None:
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")

    def _check_num_partitions_correctness_if_needed(self) -> None:
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of samples in the dataset.")

    def _initialize_alpha(self, alpha: typing.Union[int, float, typing.List[float], npt.NDArray[np.float_]]) -> \
    npt.NDArray[np.float_]:
        if isinstance(alpha, int):
            alpha = np.array([float(alpha)], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, float):
            alpha = np.array([alpha], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, typing.List):
            if len(alpha) != self._num_partitions:
                raise ValueError("If passing alpha as a List, it needs to be of length of equal to num_partitions.")
            alpha = np.asarray(alpha)
        elif isinstance(alpha, np.ndarray):
            if alpha.ndim == 1 and alpha.shape[0] != self._num_partitions:
                raise ValueError(
                    "If passing alpha as an NDArray, its length needs to be of length equal to num_partitions.")
            elif alpha.ndim == 2:
                alpha = alpha.flatten()
                if alpha.shape[0] != self._num_partitions:
                    raise ValueError(
                        "If passing alpha as an NDArray, its size needs to be of length equal to num_partitions.")
        else:
            raise ValueError("The given alpha format is not supported.")
        if not (alpha > 0).all():
            raise ValueError(f"Alpha values should be strictly greater than zero. Instead it'd be converted to {alpha}")
        return alpha

    def _determine_partition_id_to_indices_if_needed(self) -> None:
        if self._partition_id_to_indices_determined:
            return

        if self.custom_partitioner_function is not None:
            self._partition_id_to_indices = self.custom_partitioner_function(self._num_partitions, len(self.dataset),
                                                                             shuffle_indices=True)
        else:
            targets = np.asarray(self.dataset.targets)
            self._unique_classes = np.unique(targets).tolist()
            assert self._unique_classes is not None
            self._avg_num_of_samples_per_partition = len(self.dataset) / self._num_partitions

            sampling_try = 0
            while True:
                partition_id_to_indices: typing.Dict[int, typing.List[int]] = {nid: [] for nid in
                                                                               range(self._num_partitions)}
                for k in self._unique_classes:
                    indices_representing_class_k = np.where(targets == k)[0]
                    class_k_division_proportions = self._rng.dirichlet(self._alpha)
                    nid_to_proportion_of_k_samples = {nid: class_k_division_proportions[nid] for nid in
                                                      range(self._num_partitions)}
                    if self._self_balancing:
                        assert self._avg_num_of_samples_per_partition is not None
                        for nid in nid_to_proportion_of_k_samples.copy():
                            if len(partition_id_to_indices[nid]) > self._avg_num_of_samples_per_partition:
                                nid_to_proportion_of_k_samples[nid] = 0
                        sum_proportions = sum(nid_to_proportion_of_k_samples.values())
                        for nid in nid_to_proportion_of_k_samples:
                            nid_to_proportion_of_k_samples[nid] /= sum_proportions
                    cumsum_division_fractions = np.cumsum(list(nid_to_proportion_of_k_samples.values()))
                    cumsum_division_numbers = cumsum_division_fractions * len(indices_representing_class_k)
                    indices_on_which_split = cumsum_division_numbers.astype(int)[:-1]
                    split_indices = np.split(indices_representing_class_k, indices_on_which_split)
                    for nid in range(self._num_partitions):
                        partition_id_to_indices[nid].extend(split_indices[nid].tolist())
                min_sample_size_on_client = min(len(indices) for indices in partition_id_to_indices.values())
                if min_sample_size_on_client >= self._min_partition_size:
                    break
                sampling_try += 1
                if sampling_try == 10:
                    raise ValueError(
                        "The max number of attempts (10) was reached. Please update the values of alpha and try again.")

            if self._shuffle:
                for indices in partition_id_to_indices.values():
                    self._rng.shuffle(indices)
            self._partition_id_to_indices = partition_id_to_indices

        self._partition_id_to_indices_determined = True

    def load_partition(self, partition_id: int) -> torch.utils.data.Dataset:
        self._determine_partition_id_to_indices_if_needed()
        indices = self._partition_id_to_indices[partition_id]
        return Subset(self.dataset, indices)


class DistributedDataset(torch.utils.data.Dataset):
    """
    As per FederatedDataset of https://github.com/adap/flower
    """

    def __init__(
            self,
            dataloader,
            train_transform=None,
            val_transform=None,
            alpha=0.5,
            num_partitions=10,
            min_partition_size=10,
            self_balancing=True,
            shuffle=False,
            seed=42,
            val_split=0.0,
            **kwargs,
    ):
        self._dataloader = dataloader
        self._dataloader_args = {'seed': seed, 'global_args': kwargs['global_args']}
        self._dataloader_args.update(kwargs)

        # 1. Instantiate Training Data (With Augmentation)
        train_args = self._dataloader_args.copy()
        train_args['transform'] = train_transform

        # We instantiate the dataset specifically for the training portion
        full_train_ds_augmented = self._dataloader(train=True, **train_args)

        # 2. Instantiate Validation Data (Deterministic)
        # We load the SAME data (train=True) but with the validation transform
        val_args = self._dataloader_args.copy()
        val_args['transform'] = val_transform
        full_train_ds_deterministic = self._dataloader(train=True, **val_args)

        # 3. Calculate Split Indices
        # We can use either instance to calculate the split indices
        train_subset_indices, val_subset_indices = get_train_val_split(full_train_ds_augmented, val_split, seed)

        # 4. Assign Subsets to the correct underlying instance

        # Train subset -> uses augmented dataset
        if isinstance(train_subset_indices, torch.utils.data.Subset):
            self.train_ds = Subset(full_train_ds_augmented, train_subset_indices.indices)
        else:
            self.train_ds = full_train_ds_augmented

        # Validation subset -> uses deterministic dataset
        if val_subset_indices is not None:
            self.val_ds = Subset(full_train_ds_deterministic, val_subset_indices.indices)
        else:
            self.val_ds = None

        # Store val_transform for testing logic
        self.val_transform = val_transform

        custom_partitioner_function = self.train_ds.dataset.custom_partitioner_function if hasattr(
            self.train_ds.dataset, 'custom_partitioner_function') else None

        # 5. Create data partitioner using ONLY the training subset
        self.partitioner = DirichletDataPartitioner(
            dataset=self.train_ds,
            num_partitions=num_partitions,
            alpha=alpha,
            min_partition_size=min_partition_size,
            self_balancing=self_balancing,
            shuffle=shuffle,
            seed=seed,
            custom_partitioner_function=custom_partitioner_function
        )

        self._seed = seed

    @property
    def num_classes(self):
        return self.train_ds.num_classes

    @property
    def classes(self):
        return list(self.train_ds.dataset._classes.keys()) if hasattr(self.train_ds, 'dataset') else list(
            self.train_ds._classes.keys())

    def __getitem__(self, idx):
        return self.train_ds[idx]

    def __len__(self):
        return len(self.train_ds)

    def load_partition(self, partition_id: int) -> torch.utils.data.Dataset:
        partition = self.partitioner.load_partition(partition_id)
        return partition

    def load_test_set(self) -> torch.utils.data.Dataset:
        # Test set uses validation (deterministic) transform
        test_args = self._dataloader_args.copy()
        test_args['transform'] = self.val_transform

        test_ds = self._dataloader(train=False, **test_args)
        return test_ds

    def load_validation_set(self) -> torch.utils.data.Dataset:
        return self.val_ds

    def get_collate_fn(self):
        # Use getattr to safely return None if custom_collate_fn is missing
        return getattr(self._dataloader, 'custom_collate_fn', None)


def available_datasets():
    return list(dataloaders.keys())


def load_data(name='cifar100', num_partitions=10, min_num_samples=10, split='iid', seed=42, global_args=None,
              val_split=0.0):
    assert name in dataloaders.keys(), 'Dataset `{}` is not available. Available datasets are `{}`'.format(name,
                                                                                                           available_datasets())

    try:
        # This handles cases where the user passes "0.1" or "10"
        alpha_val = float(split)
        alpha = alpha_val
    except ValueError:
        # If it's not a number, check the keywords
        if split == 'iid':
            alpha = 100000.0
        elif split == 'noniid':
            alpha = 0.5
        else:
            # Properly RAISE the error so the script stops
            raise ValueError(
                f'split must be "iid", "noniid", or a number. Received: {split}'
            )

    ds, folder_name = dataloaders[name]

    # audio_datasets = ['audioset', 'speechcommands']

    if name == 'speechcommands':
        train_transform = get_speechcommands_transforms(train=True)
        val_transform = get_speechcommands_transforms(train=False)
    else:
        train_transform = get_default_vit_transforms(train=True)
        val_transform = get_default_vit_transforms(train=False)

    return DistributedDataset(
        dataloader=ds,
        train_transform=train_transform,
        val_transform=val_transform,
        alpha=alpha,
        num_partitions=num_partitions,
        min_partition_size=min_num_samples,
        self_balancing=False,
        shuffle=True,
        seed=seed,
        global_args=global_args,
        val_split=val_split
    )