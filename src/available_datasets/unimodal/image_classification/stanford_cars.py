import os
import ssl
import torchvision



ssl._create_default_https_context = ssl._create_unverified_context


class StanfordCars(torchvision.datasets.StanfordCars):
    custom_collate_fn = None

    def __init__(self, train=False, transform=None, **kwargs):
        path = os.path.join(os.environ['TORCH_DATA_DIR'], 'stanfordcars')
        super().__init__(path,
                         split="train" if train else "test",
                         transform=transform,
                         target_transform=None,
                         download=False)

        if self.transform is None:
            self.transform = torchvision.models.ViT_B_16_Weights.DEFAULT.transforms()

        self.data_processor = None

    @property
    def num_classes(self):
        return 196

    @property
    def targets(self):
        return [sample[1] for sample in self._samples]

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        return image, label

    def __len__(self):
        return len(self._samples)