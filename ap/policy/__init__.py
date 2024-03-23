import torch
from tianshou.data import Batch

from ap.policy.div import DivPPOPolicy
from ap.policy.imap import IMAPPPOPolicy
from ap.policy.ppo import BasePolicy, PPOPolicy


class RandomPolicy(BasePolicy):
    def forward(self, batch, state=None, **kwargs):
        act = torch.rand(len(batch), self.act_dim) * 2 - 1
        return Batch(act=act)

    def learn(self, batch, **kwargs):
        pass


class ZeroPolicy(BasePolicy):
    def forward(self, batch, state=None, **kwargs):
        act = torch.zeros(len(batch), self.act_dim)
        return Batch(act=act)

    def learn(self, batch, **kwargs):
        pass


class DensePPOPolicy(PPOPolicy):
    def __init__(self, dense_reward_type="original", **kwargs):
        super().__init__(**kwargs)
        self.dense_reward_type = dense_reward_type
        assert dense_reward_type is not None, "dense_reward_type is not specified"
        self._set_critic(dense_reward_type)

    def _get_adv(self, batch, idx):
        return super()._get_adv(batch, idx, self.dense_reward_type)


POLICY_DICT = {
    "base": PPOPolicy,
    "imap": IMAPPPOPolicy,
    "dense": DensePPOPolicy,
    "random": RandomPolicy,
    "zero": ZeroPolicy,
    "div": DivPPOPolicy,
}
