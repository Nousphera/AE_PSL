import os
import ssl
import torch
import torchvision


ssl._create_default_https_context = ssl._create_unverified_context

from PIL import ImageFile  # Add this import

# Tell PIL to load truncated images without throwing an OSError
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SUN397(torchvision.datasets.SUN397):
    custom_collate_fn = None

    def __init__(self, train=False, transform=None, **kwargs):

        path = os.path.join(os.environ['TORCH_DATA_DIR'], 'sun397')
        super().__init__(path,
                         transform=transform,
                         target_transform=None,
                         download=True)

        if self.transform is None:
            self.transform = torchvision.models.ViT_B_16_Weights.DEFAULT.transforms()

        self.data_processor = None

        # Deterministic 80/20 Train/Test split
        # We use a hardcoded seed here to guarantee the test set matches the held-out train data
        # regardless of when or where the dataloader is instantiated.
        generator = torch.Generator().manual_seed(42)
        dataset_len = len(self._image_files)
        shuffled_indices = torch.randperm(dataset_len, generator=generator).tolist()

        train_len = int(0.8 * dataset_len)

        if train:
            indices = shuffled_indices[:train_len]
        else:
            indices = shuffled_indices[train_len:]

        # Overwrite the internal lists used by torchvision's __getitem__
        self._image_files = [self._image_files[i] for i in indices]
        self._labels = [self._labels[i] for i in indices]

    @property
    def num_classes(self):
        return 397

    @property
    def targets(self):
        return self._labels

    def __getitem__(self, idx):
        # super().__getitem__ will now naturally pull from our filtered subset
        image, label = super().__getitem__(idx)

        return image, label

    def __len__(self):
        return len(self.targets)