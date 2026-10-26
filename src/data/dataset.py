import torch
from torch.utils.data import Dataset


class TumorSegmentationDataset(Dataset):

    def __init__(
        self,
        image_patches,
        label_patches
    ):
        self.image_patches = image_patches
        self.label_patches = label_patches

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, index):
        image = self.image_patches[index]
        label = self.label_patches[index]

        image_tensor = torch.tensor(
            image,
            dtype=torch.float32
        )

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor, label_tensor