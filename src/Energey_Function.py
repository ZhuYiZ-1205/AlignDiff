import torch
import torch.nn as nn
import torch.nn.functional as F

class DeeperSequenceEnergyFunction(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1, num_layers=4, dropout_prob=0.2):
        super(DeeperSequenceEnergyFunction, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ELU())
        layers.append(nn.Dropout(dropout_prob))

        for _ in range(num_layers - 2):  
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        out = self.network(x)
        return out.squeeze(-1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
