from typing import Callable, Tuple, Any
from torch.utils.data import Dataset

class EuroSAT(Dataset):
    """
    Thin wrapper that lets you plug any (image, label) dataset into a standard
    PyTorch Dataset with optional transforms.
    """
    def __init__(self, dataset, transform: Callable | None = None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img, y = self.dataset[index]
        x = self.transform(img) if self.transform else img
        return x, y

    def __len__(self) -> int:
        return len(self.dataset)
