from torch.utils.data import Dataset
from helpers import *
import torch

    
class LoadData(Dataset):
    def __init__(self):
        super(LoadData, self).__init__()
        self.batch = []
        self.num = [binary(i, 15) for i in range(1, pow(2, 15) - 1, 2)]
        self.n = len(self.num)
        self.label = []
        for i in range(self.n):
            self.batch.append(gauss(mean=0, std=1, n=128))
            self.label.append([1])


    def __getitem__(self, index):
        return torch.Tensor(self.batch[index]), torch.Tensor(self.num[index]), torch.Tensor(self.label[index])

    def __len__(self):
        return len(self.batch)