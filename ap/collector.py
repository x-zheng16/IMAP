import time
from collections import defaultdict

import numpy as np
import tianshou
import torch
from tianshou.data import Batch, to_numpy
from tianshou.policy import BasePolicy

from ap.util.utils import time_spent


class Collector(tianshou.data.Collector):
    def __init__(
        self, policy, env, buffer=None, preprocess_fn=None, exploration_noise=False
    ):
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)
        self.last_rew = self.last_len = self.last_success_rate = 0.0

    def collect(
        self,
        n_step=None,
        n_episode=None,
        random=False,
        render=None,
        gym_reset_kwargs=None,
    ):
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[: min(self.env_num, n_episode)]
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []
        final_state = defaultdict(list)
        success_list = []

        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                try:
                    act_sample = [self._action_space[i].sample() for i in ready_env_ids]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                assert isinstance(self.policy, BasePolicy), "Unsupported policy type"
                with torch.no_grad():
                    result = self.policy(self.data, last_state)

                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            result = self.env.step(action_remap, ready_env_ids)  # type: ignore
            if len(result) == 5:
                obs_next, rew, terminated, truncated, info = result
                done = np.logical_or(terminated, truncated)
            elif len(result) == 4:
                obs_next, rew, done, info = result
                if isinstance(info, dict):
                    truncated = info["TimeLimit.truncated"]
                else:
                    truncated = np.array(
                        [
                            info_item.get("TimeLimit.truncated", False)
                            for info_item in info
                        ]
                    )
                terminated = np.logical_and(done, ~truncated)
            else:
                raise ValueError()

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info,
            )
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                        act=self.data.act,
                    )
                )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(self.data, ready_env_ids)

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                for _done, _info in zip(done, info):
                    if _done is True:
                        if "success" in _info:
                            success_list.append(_info["success"])

                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                self._reset_env_with_ids(
                    env_ind_local, env_ind_global, gym_reset_kwargs
                )
                for i in env_ind_local:
                    self._reset_state(i)

                # record useful final info
                info = self.data.info[env_ind_local]
                if "state" in info:
                    for k, v in info["state"].items():
                        if k in ["x", "vx", "y", "vy", "z", "vz"]:
                            final_state[k] += v.tolist()
                else:
                    for k in ["x_position", "x_velocity"]:
                        if k in info:
                            final_state[k[:3]] += info[k].tolist()

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if (n_step and step_count >= n_step) or (
                n_episode and episode_count >= n_episode
            ):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={},
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = map(
                np.concatenate, [episode_rews, episode_lens, episode_start_indices]
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
            final_state = {k: f"{np.mean(v):5.2f}" for k, v in final_state.items()}
            final_state = {k: final_state[k] for k in sorted(final_state)}
            self.last_rew, self.last_len = rew_mean, len_mean
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean, len_mean = self.last_rew, self.last_len
            rew_std = len_std = 0

        success_rate = (
            np.mean(success_list) if len(success_list) else self.last_success_rate
        )
        self.last_success_rate = success_rate
        collect_result = {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": " ".join(f"{x:.2f}" for x in rews),
            "lens": " ".join(map(str, lens)),
            "idxs": " ".join(map(str, idxs)),
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
            "success": " ".join(map(str, success_list)),
            "success_rate": success_rate,
            "fps": int(step_count / (time.time() - start_time)),
            "time_spent": time_spent(time.time() - start_time),
            "final_state": final_state,
        }
        return collect_result
