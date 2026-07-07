import os
import torchaudio
from torch.utils.data import Dataset


class SpeechCommandsV2(Dataset):
    def __init__(self, train: bool, transform=None, global_args=None, seed=None):
        self.transform = transform
        self.data_dir = os.environ.get('TORCH_DATA_DIR', '../../shared_data/datasets')

        subset = "training" if train else "testing"
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=self.data_dir,
            url='speech_commands_v0.02',
            folder_in_archive='SpeechCommands',
            download=True,
            subset=subset
        )

        self.num_classes = 35

        # Hardcode the known 35 classes for SpeechCommands V2 to avoid the first iteration entirely
        classes = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go',
                   'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
                   'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
        self._classes = {label: i for i, label in enumerate(sorted(classes))}

        # Extract targets directly from the file paths in the _walker attribute
        # path format is usually: .../SpeechCommands/speech_commands_v0.02/<label>/<filename>.wav
        self.targets = []
        for file_path in self.dataset._walker:
            # Extract the parent directory name, which is the label
            label = os.path.basename(os.path.dirname(str(file_path)))
            self.targets.append(self._classes[label])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[idx]
        target = self._classes[label]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, target