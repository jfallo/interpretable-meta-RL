import torch
import torch.nn as nn


class BatchedMLPs(nn.Module):
    def __init__(self, k, sizes):
        super().__init__()
        self.W, self.b = nn.ParameterList(), nn.ParameterList()
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            linear_layers = [nn.Linear(in_size, out_size) for _ in range(k)]
            self.W.append(nn.Parameter(torch.stack([l.weight.T for l in linear_layers])))
            self.b.append(nn.Parameter(torch.stack([l.bias for l in linear_layers]).unsqueeze(1)))

    def forward(self, x):
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = torch.baddbmm(b, x, W)
            if i < len(self.W) - 1:
                x = torch.relu(x)
        
        return x

def build_update_MLPs(num_MLPs, input_size, output_size, hidden_size= 5):
    return BatchedMLPs(num_MLPs, [input_size, 
                                  hidden_size, 
                                  hidden_size, 
                                  hidden_size, 
                                  output_size])

def build_choice_MLP(input_size, output_size, hidden_size= 5):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )