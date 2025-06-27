from torch.utils.data import Dataset


class ModalityData(NamedTuple):
    features: torch.Tensor; lengths: torch.Tensor; name: str

class BimodalDataset(Dataset):
    def __init__(self, mod_a: ModalityData, mod_b: ModalityData, labels: torch.Tensor):
        assert len(mod_a.features) == len(mod_b.features) == len(labels)
        self.mod_a, self.mod_b, self.labels = mod_a, mod_b, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (self.mod_a.features[idx], self.mod_a.lengths[idx],
                self.mod_b.features[idx], self.mod_b.lengths[idx],
                self.labels[idx])

class TrimodalDataset(Dataset):
    def __init__(self, mod_t: ModalityData, mod_a: ModalityData, mod_v: ModalityData, labels: torch.Tensor):
        assert len(mod_t.features) == len(mod_a.features) == len(mod_v.features) == len(labels)
        self.mod_t, self.mod_a, self.mod_v, self.labels = mod_t, mod_a, mod_v, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (self.mod_t.features[idx], self.mod_t.lengths[idx], self.mod_a.features[idx], self.mod_a.lengths[idx], self.mod_v.features[idx], self.mod_v.lengths[idx], self.labels[idx])
