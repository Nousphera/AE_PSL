import os
import ssl
import PIL
import torchvision


ssl._create_default_https_context = ssl._create_unverified_context


class Food101(torchvision.datasets.Food101):
    custom_collate_fn = None

    def __init__(self, train=False, transform=None, **kwargs):
        # We pass 'transform' to super, but we also ensure we don't overwrite it below.
        super().__init__(os.path.join(os.environ['TORCH_DATA_DIR'], 'food101'),
                         split="train" if train else "test",
                         transform=transform,
                         target_transform=None,
                         download=True)

        if self.transform is None:
            self.transform = torchvision.models.ViT_B_16_Weights.DEFAULT.transforms()


        self.data_processor = None

    @property
    def num_classes(self):
        return 101

    @property
    def targets(self):
        return self._labels

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.targets)