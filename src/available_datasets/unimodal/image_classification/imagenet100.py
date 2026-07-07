import json
import os
import ssl

import torchvision


# Disable SSL verification for consistency
ssl._create_default_https_context = ssl._create_unverified_context


class ImageNet100(torchvision.datasets.ImageFolder):
    custom_collate_fn = None

    def __init__(self, train=False, transform=None, **kwargs):
        # Locate the dataset root
        root_dir = os.path.join(os.environ['TORCH_DATA_DIR'], 'imagenet100')

        split_dir = 'train' if train else 'test'

        full_path = os.path.join(root_dir, split_dir)

        if not os.path.exists(full_path):
            raise ValueError(f"Directory not found: {full_path}")

        super().__init__(root=full_path, transform=transform, target_transform=None)

        self.transform = transform
        if self.transform is None:
            self.transform = torchvision.models.ViT_B_32_Weights.DEFAULT.transforms()

        self.data_processor = None

        # Load Labels.json to map folder names (class IDs) to human-readable strings
        # This is stored for reference, though training uses the folder-derived indices.
        labels_path = os.path.join(root_dir, 'Labels.json')
        self.class_label_map = {}
        if os.path.exists(labels_path):
            try:
                with open(labels_path, 'r') as f:
                    self.class_label_map = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load Labels.json: {e}")

    def __getitem__(self, idx):
        """
        Overrides ImageFolder.__getitem__ to return the dictionary format
        required by the Data2Seq model wrapper.
        """
        path, target = self.samples[idx]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    @property
    def num_classes(self):
        return len(self.classes)