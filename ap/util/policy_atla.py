import torch as torch
import torch.nn as nn

HIDDEN_SIZES = (64, 64)
ACTIVATION = nn.Tanh


class CtsPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=HIDDEN_SIZES):
        super().__init__()
        self.activation = ACTIVATION()
        self.action_dim = action_dim

        self.affine_layers = nn.ModuleList()
        prev_size = state_dim
        for i in hidden_sizes:
            lin = nn.Linear(prev_size, i, bias=True)
            self.affine_layers.append(lin)
            prev_size = i
        self.final_mean = nn.Linear(prev_size, action_dim, bias=True)
        stdev_init = torch.zeros(action_dim)
        self.log_stdev = torch.nn.Parameter(stdev_init)

    def forward(self, x, state=None):
        x = torch.as_tensor(x, device="cuda", dtype=torch.float32)
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        means = self.final_mean(x)
        std = torch.exp(self.log_stdev)
        return means, std, None


class CtsLSTMPolicy(CtsPolicy):
    def __init__(self, state_dim, action_dim, hidden_sizes=HIDDEN_SIZES):
        super().__init__(state_dim, action_dim, hidden_sizes)
        self.hidden_sizes = hidden_sizes

        self.embedding_layer = nn.Linear(state_dim, self.hidden_sizes[0])
        self.lstm = nn.LSTM(*self.hidden_sizes, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_sizes[-1], action_dim)

    def forward(self, x, state=None):
        self.lstm.flatten_parameters()
        x = torch.as_tensor(x, device="cuda", dtype=torch.float32)
        embedding = self.embedding_layer(x).unsqueeze(-2)
        output, state = self.lstm(embedding, state)
        means = self.output_layer(output[:, -1])
        std = torch.exp(self.log_stdev)
        return means, std, state


POLICY_NETS = {"CtsPolicy": CtsPolicy, "CtsLSTMPolicy": CtsPolicy}
