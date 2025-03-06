import torch
import torch.nn as nn
import torch.optim as optim

class ProteinNet(nn.Module):
    def __init__(self, fea_dim, hidden_dim1, hidden_dim2, num_layers):
        super(ProteinNet, self).__init__()

        self.layers1 = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(fea_dim, fea_dim),
                nn.ReLU()
            )
            for _ in range(num_layers)
        ])

        self.layers3 = nn.Linear(fea_dim, hidden_dim1)
        self.layers4 = nn.Linear(hidden_dim1, hidden_dim2)
        self.predict = nn.Linear(hidden_dim2, 1)

    def forward(self, mw):
        mw = self.layers1(mw)
        mw = self.layers3(mw)
        mw = self.layers4(mw)
        ddg = self.predict(mw)
        return ddg
