import copy
from collections import defaultdict

import numpy as np
import tianshou
import torch
from tianshou.data import Batch
from tianshou.policy.base import _gae_return
from tianshou.utils import MovAvg, RunningMeanStd
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from ap.conf.config import RepConf
from ap.conf.ppo_conf import BasePPOConf as BConf
from ap.util.utils import grad_monitor, last_layer_init, split_batch, weight_init


class BasePolicy(tianshou.policy.BasePolicy):
    def __init__(
        self,
        observation_space=None,
        action_space=None,
        action_scaling=BConf.action_scaling,
        action_bound_method=BConf.action_bound_method,
        lr_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            action_scaling,
            action_bound_method,
            lr_scheduler,
        )
        self.obs_shape = self.observation_space.shape
        self.obs_dim = np.prod(self.obs_shape)
        self.act_shape = self.action_space.shape or self.action_space.n
        self.act_dim = np.prod(self.act_shape)


class PPOPolicy(BasePolicy):
    def __init__(
        self,
        encoder,
        actor,
        critic,
        optim,
        dist_fn,
        use_rep=RepConf.use_rep,
        mlp_hidden_dim=RepConf.mlp_hidden_dim,
        obs_rep_dim=RepConf.obs_rep_dim,
        mlp_norm=RepConf.mlp_norm,
        s="PConst",
        rf_rate=BConf.rf_rate,
        ex_rate=BConf.ex_rate,
        lag_rate=BConf.lag_rate,
        total_updates=None,
        name="agent",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.actor = actor
        self.optim = optim
        self.dist_fn = dist_fn
        self.name = name

        # ppo setting
        self._eps_clip = kwargs.get("eps_clip", BConf.eps_clip)
        self._norm_adv = kwargs.get("norm_adv", BConf.norm_adv)
        self._recompute_adv = kwargs.get("rc_adv", BConf.rc_adv)
        self._grad_norm = kwargs.get("max_grad_norm", BConf.max_grad_norm)
        self._lambda = kwargs.get("gae_lambda", BConf.gae_lambda)
        self._gamma = kwargs.get("discount_factor", BConf.discount_factor)
        self._norm_return = kwargs.get("norm_return", BConf.norm_return)
        self._deterministic_eval = kwargs.get(
            "deterministic_eval", BConf.deterministic_eval
        )
        self._eps = 1e-8

        # learning info
        self.device = actor.device
        self.lr = 0 if optim is None else optim.defaults["lr"]
        self.minibatch_size = kwargs.get("minibatch_size", None)
        self.learn_info = defaultdict(list)
        self.use_rep = use_rep

        # critics
        self.critics = {"ex": critic}
        self.opts = {"ex_critic": self._get_opt(self.critics["ex"])}
        self.mse = nn.MSELoss(reduction="none")
        self.ret_rmss = defaultdict(RunningMeanStd)

        # encoder
        self.encoder = encoder
        self.opts["encoder"] = self._get_opt(self.encoder)

        # rep
        self.obs_rep_dim = obs_rep_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_norm = mlp_norm
        print(f"{self.name} | obs_shape: {self.obs_shape} | act_dim: {self.act_dim}")

        # init
        self.n_update = 0
        self.progress = 0.0
        self.total_updates = total_updates
        self.last_avg_ex_rew = MovAvg(1)
        self.lag_coef = 0
        self.in_rew_coef = 0

        # balancing config
        self.in_rew_schedule = s
        self.rf_rate = rf_rate
        self.ex_rate = ex_rate
        self.lag_rate = lag_rate

    def _set_critic(self, rew_type):
        self.critics[rew_type] = copy.deepcopy(self.critics["ex"])
        self.critics[rew_type].apply(weight_init)
        last_layer_init(self.critics[rew_type])
        self.opts[rew_type + "_critic"] = self._get_opt(self.critics[rew_type])

    def forward(self, batch, state=None, **kwargs):
        ac_input = self._get_ac_input(batch.obs)
        return self._get_actor_output(ac_input, state, **kwargs)

    def _get_obs_rep(self, obs):
        return self.encoder(self._to_torch(obs))

    def _get_ac_input(self, obs):
        return self._get_obs_rep(obs) if self.use_rep else self._to_torch(obs)

    def _logits_to_dist(self, logits):
        if isinstance(logits, tuple) or isinstance(logits, list):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        return dist

    def _get_actor_output(self, ac_input, state=None, **kwargs):
        logits, state = self.actor(ac_input, state=state)
        dist = self._logits_to_dist(logits)
        if (self._deterministic_eval and not self.training) or kwargs.get(
            "deterministic", False
        ):
            if self.action_type == "discrete":
                act = logits.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()
        logits = {str(i): logits[i] for i in range(len(logits))}
        return Batch(act=act, state=state, dist=dist, policy=Batch(logits=logits))

    # process before learn
    def process_fn(self, batch, buffer, indices):
        self.learn_info.clear()
        self._buffer, self._indices, self._batch_size = buffer, indices, len(batch)
        batch.v_s, batch.returns, batch.advs = {}, {}, {}
        batch.act = self._to_torch(batch.act)
        if len(self.obs_shape) == 1:  # state-based
            batch.obs = self._to_torch(batch.obs)
            batch.obs_next = self._to_torch(batch.obs_next)
        if "state" in batch.info:
            for k, v in batch.info["state"].items():
                batch.info["state"][k] = self._to_torch(v)
        if "rew_ex" not in batch.info:
            batch.info["rew_ex"] = batch.rew
        old_log_prob = self._get_empty()
        with torch.no_grad():
            for idx in self._split_batch():
                old_log_prob[idx] = self(batch[idx]).dist.log_prob(batch.act[idx])
        batch.logp_old = old_log_prob
        return batch

    def learn(self, batch, minibatch_size, repeat, **kwargs):
        self.repeat = repeat
        learn_info = self.learn_info
        learn_info["batch"] = batch
        self._get_intrinsic_rew(batch)
        self._compute_returns(batch, set(self.critics))
        self._get_intrinsic_rew_coef(batch, **kwargs)
        for step in range(repeat):
            if self._recompute_adv and step:
                self._compute_returns(batch, set(self.critics))
            for idx in self._split_batch(minibatch_size, shuffle=True):
                self._update_intrinsic_module(batch, idx)
                with torch.no_grad():
                    ac_input = self._get_ac_input(batch.obs[idx])
                self._learn_actor(batch, idx, ac_input)
                self._learn_critics(batch, idx, ac_input)
        self.n_update += 1
        self.progress = min(self.n_update / self.total_updates, 1)
        learn_info["progress"] = self.progress
        return learn_info

    def _get_intrinsic_rew(self, batch):
        pass

    def _get_intrinsic_rew_coef(self, batch, **kwargs):
        last_avg_ex_rew = self.last_avg_ex_rew.get()
        self.learn_info["last_avg_ex_rew"] = last_avg_ex_rew
        if "success" in batch.info:
            ex_rew = batch.info["rew_ex"][batch.done]
            avg_ex_rew = np.mean(ex_rew) if len(ex_rew) else last_avg_ex_rew
            expected_avg_ex_rew = min(self.ex_rate * last_avg_ex_rew, 1)
        else:
            avg_ex_rew = batch.info["rew_ex"].mean()
            if last_avg_ex_rew == 0:
                last_avg_ex_rew = avg_ex_rew
            if last_avg_ex_rew > 0:
                expected_avg_ex_rew = last_avg_ex_rew * self.ex_rate
            else:
                expected_avg_ex_rew = last_avg_ex_rew / self.ex_rate
        dlag = avg_ex_rew - expected_avg_ex_rew
        self.lag_coef = max(self.lag_coef - self.lag_rate * dlag, 0)
        self.last_avg_ex_rew.add(avg_ex_rew)

        if self.in_rew_schedule == "L":
            self.in_rew_coef = 1 / (1 + self.lag_coef)
        elif self.in_rew_schedule == "PLinear":
            self.in_rew_coef = 1 - self.progress
        elif self.in_rew_schedule == "PConst":
            self.in_rew_coef = 1
        elif self.in_rew_schedule == "PExp":
            self.in_rew_coef = 0.001**self.progress
        self.learn_info["lag_coef"] = self.lag_coef
        self.learn_info["in_rew_coef"] = self.in_rew_coef

    def _update_intrinsic_module(self, batch, idx):
        pass

    def _learn_actor(self, batch, idx, ac_input):
        dist = self._get_actor_output(ac_input).dist
        adv = self._normalize_adv(self._get_adv(batch, idx))
        ratio = (dist.log_prob(batch.act[idx]) - batch.logp_old[idx]).exp().float()
        surr1 = ratio * adv
        surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * adv
        clip_loss = -torch.min(surr1, surr2).mean()

        self.optim.zero_grad()
        clip_loss.backward()
        self._clip_grad(self.actor)
        self.optim.step()

        learn_info = self.learn_info
        learn_info["loss/actor"].append(clip_loss.item())
        learn_info["grad/actor"].append(grad_monitor(self.actor))
        with torch.no_grad():
            approx_kl = (ratio - 1) - ratio.log()
            clip_frac = ((ratio - 1).abs() > self._eps_clip).float().mean()
        learn_info["max_ratio"].append(ratio.max().item())
        learn_info["max_kl"].append(approx_kl.max().item())
        learn_info["mean_kl"].append(approx_kl.mean().item())
        learn_info["clip_frac"].append(clip_frac.item())

    def _learn_critics(self, batch, idx, ac_input):
        for k in set(self.critics):
            vf_loss = self.mse(self.critics[k](ac_input), batch.returns[k][idx]).mean()

            self.opts[k + "_critic"].zero_grad()
            vf_loss.backward()
            self._clip_grad(self.critics[k])
            self.opts[k + "_critic"].step()

            learn_info = self.learn_info
            learn_info["loss/critic/" + k].append(vf_loss.item())
            learn_info["grad/critic/" + k].append(grad_monitor(self.critics[k]))

    def _get_adv(self, batch, idx, rew_type=None):
        adv_ex = batch.advs["ex"][idx]
        self.learn_info["abs_adv_ex"] = adv_ex.abs().mean().item()
        if rew_type is None:
            assert self.in_rew_schedule == "EX", "Use EX when no intrinsic bonus!"
            return adv_ex
        adv_in = batch.advs[rew_type][idx]
        self.learn_info["abs_adv_in"] = adv_in.abs().mean().item()
        if self.in_rew_schedule == "RF":
            return adv_in if self.progress < self.rf_rate else adv_ex
        else:
            assert self.in_rew_schedule in ["L", "PConst", "PLinear", "PExp"]
            return adv_ex + self.in_rew_coef * adv_in

    def _normalize_adv(self, adv):
        return (adv - adv.mean()) / adv.std() if self._norm_adv else adv

    def _to_torch(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _to_numpy(self, x):
        return x.detach().cpu().numpy()

    def _get_empty(self, *shape):
        return torch.empty([self._batch_size, *shape], device=self.device)

    def _split_batch(self, minibatch_size=None, batch_size=None, shuffle=False):
        return split_batch(
            minibatch_size or self.minibatch_size,
            batch_size or self._batch_size,
            shuffle,
        )

    def _clip_grad(self, net):
        if self._grad_norm is not None:
            clip_grad_norm_(net.parameters(), self._grad_norm)

    def _get_opt(self, net, lr_scale=1):
        return torch.optim.Adam(net.parameters(), self.lr / lr_scale)

    def _compute_returns(self, batch, critics_name):
        v_s = defaultdict(self._get_empty)
        v_s_next = defaultdict(self._get_empty)
        with torch.no_grad():
            for idx in self._split_batch():
                ac_input = self._get_ac_input(batch.obs[idx])
                ac_input_next = self._get_ac_input(batch.obs_next[idx])
                for k in critics_name:
                    v_s[k][idx] = self.critics[k](ac_input)
                    v_s_next[k][idx] = self.critics[k](ac_input_next)
        batch.v_s = v_s
        for k in critics_name:
            batch.returns[k], batch.advs[k] = self.__compute_returns(
                batch,
                v_s_next[k].cpu().numpy(),
                v_s[k].cpu().numpy(),
                self.ret_rmss[k],
                batch.info["rew_" + k],
            )
            xvar = 1 - (batch.returns[k] - batch.v_s[k]).var() / (
                batch.returns[k].var() + 1e-8
            )
            self.learn_info["xvar/" + k].append(xvar.item())

    def __compute_returns(self, batch, v_s_next, v_s, ret_rms, rew):
        if self._norm_return:
            v_s = v_s * np.sqrt(ret_rms.var + self._eps)
            v_s_next = v_s_next * np.sqrt(ret_rms.var + self._eps)

        unnormalized_returns, advantages = self.compute_episodic_return(
            batch, v_s_next, v_s, rew
        )
        if self._norm_return:
            returns = unnormalized_returns / np.sqrt(ret_rms.var + self._eps)
            ret_rms.update(unnormalized_returns)
        else:
            returns = unnormalized_returns
        returns = self._to_torch(returns)
        advantages = self._to_torch(advantages)
        return returns, advantages

    def compute_episodic_return(self, batch, v_s_next, v_s, rew):
        buffer, indices = self._buffer, self._indices
        v_s_next = v_s_next * BasePolicy.value_mask(buffer, indices)
        end_flag = batch.done.copy()
        end_flag[np.isin(indices, buffer.unfinished_index())] = True
        advantage = _gae_return(v_s, v_s_next, rew, end_flag, self._gamma, self._lambda)
        returns = advantage + v_s
        return returns, advantage
