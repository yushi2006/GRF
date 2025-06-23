from torch.utils.data import Dataset


class UniModalDataset(Dataset):
    def __init__(self, modality):
        self.modality = modality

    def __len__(self):
        return len(self.modality)

    def __getitem__(self, idx: int):
        return self.modality[idx]
