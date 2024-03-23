import copy

import torch
from tianshou.data import to_numpy
from torch.distributions import kl_divergence

from ap.policy.ppo import PPOPolicy
from ap.util.utils import grad_monitor


class DivPPOPolicy(PPOPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mimic = copy.deepcopy(self.actor)
        self.opts["mimic"] = self._get_opt(self.mimic, 3)
        self._set_critic("div")

    def _get_mimic_dist(self, obs_rep):
        logits, _ = self.mimic(obs_rep, state=None)
        return self._logits_to_dist(logits)

    def _get_intrinsic_rew(self, batch):
        rew_div = self._get_empty()
        with torch.no_grad():
            for idx in self._split_batch():
                logits = batch.policy.logits[idx]
                old_dist = self._logits_to_dist(
                    [logits[str(i)] for i in range(len(logits.keys()))]
                )
                mimic_dist = self._get_mimic_dist(self._get_ac_input(batch.obs[idx]))
                rew_div[idx] = kl_divergence(old_dist, mimic_dist)
        batch.info["rew_div"] = to_numpy(rew_div)
        self.learn_info["rew_intrinsic"] = batch.info["rew_div"].mean()

    def _update_intrinsic_module(self, batch, idx):
        learn_info = self.learn_info
        logits = batch.policy.logits[idx]
        old_dist = self._logits_to_dist(
            [logits[str(i)] for i in range(len(logits.keys()))]
        )
        mimic_dist = self._get_mimic_dist(self._get_ac_input(batch.obs[idx]))
        mimic_loss = kl_divergence(old_dist, mimic_dist).mean()

        self.opts["mimic"].zero_grad()
        mimic_loss.backward()
        self._clip_grad(self.mimic)
        self.opts["mimic"].step()

        learn_info["loss/mimic"].append(mimic_loss.item())
        learn_info["grad/mimic"].append(grad_monitor(self.mimic))

    def _get_adv(self, batch, idx):
        return super()._get_adv(batch, idx, "div")
