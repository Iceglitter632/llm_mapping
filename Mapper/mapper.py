import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

class TokenMapper(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.model = nn.Linear(input_dim, output_dim)
        self.model.to(args.device)

    def forward(self, one_hot_token):
        return self.model(one_hot_token)