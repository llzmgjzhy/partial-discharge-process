from torch.utils.data import Dataset


class AItrainDataset(Dataset):
    def __init__(self, content_array, label_array):
        super(AItrainDataset, self).__init__()
        self.content_array = content_array
        self.label_array = label_array

    def __getitem__(self, index):
        return self.content_array[index], self.label_array[index]

    def __len__(self):
        return len(self.content_array)
