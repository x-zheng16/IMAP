import json

import numpy as np
from gym.spaces import Box

from ap import WORKPLACE
from ap.agent import load_agent
from ap.envs.env_mujoco import MuJoCoVenvWrapper, get_mujoco_env_f
from ap.envs.venvs import ShmemVectorEnv


def set_attacker_reward(env_name, info, is_black_box, vx_ratio):
    for i in range(len(info)):
        if "success" in info[i]:
            info[i]["rew_ex"] = 1 - info[i]["success"]
        else:
            if is_black_box:
                vx = info[i]["state"]["vx"]
                if "HalfCheetah" in env_name or "Ant" in env_name:
                    info[i]["rew_ex"] = -vx
                else:
                    info[i]["rew_ex"] = -vx_ratio * vx - 1
            else:
                info[i]["rew_ex"] = -info[i]["rew_dense"]


class MuJoCoObsAttackVecEnv(MuJoCoVenvWrapper):
    def __init__(self, venv, victim, epsilon=0.01, **kwargs):
        super().__init__(venv, victim, **kwargs)
        self.epsilon = epsilon
        # print(f"vx_ratio is {self.vx_ratio}")

    def __getattribute__(self, key):
        if key == "action_space":
            act_shape = self.observation_space[0].shape
            act_space = Box(low=-1, high=1, shape=act_shape)
            return [act_space for _ in range(len(self.venv))]
        else:
            return super().__getattribute__(key)

    def step(self, act, id=None):
        id = self.venv._wrap_id(id)
        # attack in observation space
        obs_v = self.victim.normalize_obs(self.victim_obs[id])
        obs_v += self.epsilon * act
        obs_v = np.clip(obs_v, -10, 10)
        act_v = self.victim.compute_action(obs_v, id)
        # step
        obs, rew, done, info = super().step(act_v, id)
        self.victim_obs[id] = obs
        set_attacker_reward(self.env_name, info, self.is_black_box, self.vx_ratio)
        return obs, rew, done, info


class MuJoCoActAttackVecEnv(MuJoCoVenvWrapper):
    def __init__(self, venv, victim, epsilon=0.01, **kwargs):
        super().__init__(venv, victim, **kwargs)
        self.epsilon = epsilon

    def step(self, act, id=None):
        id = self.venv._wrap_id(id)
        # attack in action space
        obs_v = self.victim.normalize_obs(self.victim_obs[id])
        act_v = self.victim.compute_action(obs_v, id)
        act_v += self.map_action(self.epsilon * act)
        # step
        obs, rew, done, info = super().step(act_v, id)
        self.victim_obs[id] = obs
        set_attacker_reward(self.env_name, info, self.is_black_box, self.vx_ratio)
        return obs, rew, done, info

    def map_action(self, act):
        low, high = self.ac_space.low, self.ac_space.high
        act = low + (high - low) * (act + 1.0) / 2.0
        return act


MUJOCO_ATTACK_ENV = {
    "mujoco_obs_attack": MuJoCoObsAttackVecEnv,
    "mujoco_act_attack": MuJoCoActAttackVecEnv,
    "mujoco_sparse_obs_attack": MuJoCoObsAttackVecEnv,
    "mujoco_sparse_act_attack": MuJoCoActAttackVecEnv,
}


def make_venv_mujoco_attack(
    env_name,
    victim_name,
    n_env=1,
    n_test_env=1,
    pid_bias=0,
    bind_core=False,
    venv_cls=ShmemVectorEnv,
    epsilon=0.1,
    **kwargs,
):
    task_type = kwargs["task_type"]
    target_agent_type = kwargs["target_agent_type"]
    vx_ratio = kwargs["vx_ratio"]
    is_black_box = kwargs.get("is_black_box", True)
    with open(f"{WORKPLACE}/zoo/{target_agent_type}/agents.json", "r") as f:
        zoo = json.load(f)
    victim_info = zoo[f"{target_agent_type}/{env_name}"][victim_name]

    env_f = get_mujoco_env_f(kwargs["task_type"], env_name, kwargs["time_limit"])
    venv = venv_cls([env_f for _ in range(n_env)], pid_bias, bind_core)
    victim = load_agent(venv, **victim_info, name="victim4train")
    venv = MUJOCO_ATTACK_ENV[task_type](
        venv, victim, epsilon, is_black_box=is_black_box, vx_ratio=vx_ratio
    )

    test_venv = venv_cls([env_f for _ in range(n_test_env)], pid_bias, bind_core)
    test_victim = load_agent(test_venv, **victim_info, name="victim4test")
    test_venv = MUJOCO_ATTACK_ENV[task_type](
        test_venv, test_victim, epsilon, is_black_box=is_black_box, vx_ratio=vx_ratio
    )
    return venv, test_venv
