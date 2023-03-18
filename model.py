from torch import nn

class Generator(nn.Module):
    def __init__(self, 
                 in_shape: int,
                 out_shape: int
                 ):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_shape, out_features=out_shape)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x

# Input: 50 - 50
class Discriminator(nn.Module):
    def __init__(self,
                 in_shape: int,
                 out_shape: int
                 ):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_shape, out_features=out_shape)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x

