import copy

import numpy as np
import torch
from gym.spaces import Box, Discrete
from tianshou.utils.net import common, continuous, discrete
from torch import nn

from ap.util.utils import MLP_NORM, weight_init


class Critic(nn.Module):
    def __init__(
        self,
        preprocess,
        device="cuda",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess
        self.last = nn.Linear(preprocess.output_dim, 1)

    def forward(self, s):
        logits, _ = self.preprocess(s)
        logits = self.last(logits)
        return logits.flatten()


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        norm="None",
        activation=nn.ReLU,
        last_activation=False,
        last_norm=False,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        assert len(dims) >= 2
        norm = MLP_NORM[norm]
        self.output_dim = dims[-1]
        net = []
        for k in range(len(dims) - 1):
            net.append(nn.Linear(dims[k], dims[k + 1]))
            if norm is not None:
                net.append(norm(dims[k + 1]))
            net.append(activation())
        if not last_activation:
            net.pop()
        if (norm is not None) and (not last_norm):
            net.pop()
        self.net = nn.Sequential(*net)
        self.apply(weight_init)

    def forward(self, obs):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs.flatten(1))


def get_encoder(
    obs_shape,
    hidden_dim=64,
    obs_rep_dim=64,
    norm="BN",
    activation=nn.ReLU,
    device="cuda",
):
    if len(obs_shape) == 1:
        mlp = MLP(
            [obs_shape[0], hidden_dim, obs_rep_dim], norm, activation, device=device
        )
        encoder = nn.Sequential(mlp).to(device)
    else:
        raise Exception("Unsupported obs_shape")
    return encoder


def get_actor_critic(ac_space, obs_dim, hidden_sizes, device):
    net_a = common.Net(
        obs_dim,
        hidden_sizes=hidden_sizes,
        activation=nn.Tanh,
        device=device,
    ).to(device)

    if isinstance(ac_space, Discrete):
        actor = discrete.Actor(
            net_a,
            ac_space.n,
            device=device,
        ).to(device)
    elif isinstance(ac_space, Box):
        actor = continuous.ActorProb(
            net_a,
            ac_space.shape,
            max_action=ac_space.high[0],
            unbounded=True,
            device=device,
        ).to(device)

    critic = Critic(copy.deepcopy(net_a), device=device).to(device)
    return actor, critic


def get_model(
    ob_space,
    ac_space,
    use_rep,
    mlp_hidden_dim,
    obs_rep_dim,
    mlp_norm,
    hidden_sizes,
    device,
):
    encoder = get_encoder(
        ob_space.shape, mlp_hidden_dim, obs_rep_dim, mlp_norm, device=device
    )
    input_dim = encoder[-1].output_dim if use_rep else int(np.prod(ob_space.shape))
    actor, critic = get_actor_critic(ac_space, input_dim, hidden_sizes, device)
    return encoder, actor, critic


if __name__ == "__main__":
    print(MLP([2, 2, 2]))
