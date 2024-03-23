import json

import gym
import mujoco_maze  # noqa: F401
import numpy as np
from gym.wrappers import FlattenObservation, TimeLimit

from ap import WORKPLACE
from ap.agent import Agent, load_agent
from ap.envs.venvs import ShmemVectorEnv
from ap.envs.wrappers import VenvWrapper
from ap.envs.wrappers.annotate_state import AnnotateState
from ap.envs.wrappers.sparse_reward import SPARSE_TASK, SparseReward
from ap.util.utils import get_env_space

ASSETS = f"{WORKPLACE}/ap/envs/wrappers/assets/"
XML_FILES = {
    "Hopper-v3": ASSETS + "hopper.xml",
    "Walker2d-v3": ASSETS + "walker2d.xml",
    "HalfCheetah-v3": ASSETS + "half_cheetah.xml",
    "Ant-v3": ASSETS + "ant.xml",
    "Humanoid-v3": ASSETS + "humanoid.xml",
}


class MuJoCoVenvWrapper(VenvWrapper):
    def __init__(self, venv, victim, **kwargs):
        super().__init__(venv)
        self.victim: Agent = victim
        self.victim_obs = None
        self.ob_space, self.ac_space = get_env_space(venv)
        self.env_name = self.spec[0].id
        self.is_black_box = kwargs.get("is_black_box", True)
        if self.is_black_box is None:
            self.is_black_box = True
        self.vx_ratio = kwargs.get("vx_ratio", 1)

    def reset(self, id=None):
        id = self.venv._wrap_id(id)
        obs = super().reset(id)
        if self.victim_obs is None:
            self.victim_obs = obs
        else:
            self.victim_obs[id] = obs
        self.victim.reset_state(id)
        return obs


class MuJoCoEvalVecEnv(MuJoCoVenvWrapper):
    def step(self, act, id=None):
        id = self.venv._wrap_id(id)
        obs_v = self.victim.normalize_obs(self.victim_obs[id])
        obs_v = np.clip(obs_v, -10, 10)
        act_v = self.victim.compute_action(obs_v, id)
        obs, rew, done, info = super().step(act_v, id)
        self.victim_obs[id] = obs
        return obs, rew, done, info


def get_mujoco_env_f(task_type, env_name, time_limit=None):
    def env_f():
        if (
            "sparse" in task_type
            and env_name in SPARSE_TASK
            and env_name.endswith("-v3")
        ):
            env = gym.make(
                env_name,
                exclude_current_positions_from_observation=False,
                # xml_file=XML_FILES[env_name],
            ).unwrapped
        else:
            env = gym.make(env_name)
        if env_name.startswith("Fetch"):
            env = FlattenObservation(env)
        # try to annotate mujoco state
        env = AnnotateState(env)
        if "sparse" in task_type:
            env = SparseReward(env)
            env = TimeLimit(env, time_limit or 500)
        return env

    return env_f


def make_venv_mujoco_eval(
    env_name,
    victim_name,
    n_env=1,
    n_test_env=1,
    pid_bias=0,
    bind_core=False,
    venv_cls=ShmemVectorEnv,
    **kwargs,
):
    target_agent_type = kwargs["target_agent_type"]
    with open(f"{WORKPLACE}/zoo/{target_agent_type}/agents.json", "r") as f:
        zoo = json.load(f)
    victim_info = zoo[f"{target_agent_type}/{env_name}"][victim_name]

    env_f = get_mujoco_env_f(kwargs["task_type"], env_name, kwargs["time_limit"])
    venv = venv_cls([env_f for _ in range(n_env)], pid_bias, bind_core)
    victim = load_agent(venv, **victim_info, agent_name="victim4train")
    venv = MuJoCoEvalVecEnv(venv, victim)

    test_venv = venv_cls([env_f for _ in range(n_test_env)], pid_bias, bind_core)
    test_victim = load_agent(test_venv, **victim_info, agent_name="victim4test")
    test_venv = MuJoCoEvalVecEnv(test_venv, test_victim)
    return venv, test_venv


def make_venv_mujoco(
    env_name,
    n_env=1,
    n_test_env=1,
    pid_bias=0,
    bind_core=False,
    venv_cls=ShmemVectorEnv,
    **kwargs,
):
    env_f = get_mujoco_env_f(kwargs["task_type"], env_name, kwargs["time_limit"])
    venv = venv_cls([env_f for _ in range(n_env)], pid_bias, bind_core)
    test_venv = venv_cls([env_f for _ in range(n_test_env)], pid_bias, bind_core)
    return venv, test_venv
