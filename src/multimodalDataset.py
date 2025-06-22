from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(self, x, y, labels):
        assert len(x) == len(y) == len(labels), "Data is not aligned correctly."

        self.x = x
        self.y = y
        self.labels = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.labels[idx]
