import os
import pickle
import sys

import numpy as np
import tensorflow as tf
import torch
from tianshou.utils import RunningMeanStd

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from dataclasses import asdict

import stable_baselines
from aprl.envs.gym_compete import (
    POLICY_STATEFUL,
    env_name_to_canonical,
    get_policy_type_for_zoo_agent,
    load_zoo_agent_params,
)
from aprl.policies import base
from stable_baselines.common.policies import MlpPolicy
from tianshou.data import Batch, to_numpy

from ap.conf.config import ActorCriticConf, RepConf
from ap.net import get_model
from ap.policy import POLICY_DICT
from ap.util.policy_atla import CtsLSTMPolicy, CtsPolicy
from ap.util.utils import get_dist_fn, get_env_space, make_session


class Agent:
    def __init__(self, policy_type, policy, obs_rms, n_env):
        self.policy_type = policy_type
        self.policy = policy
        self.policy.eval()
        self.obs_rms = obs_rms or RunningMeanStd()
        self.state = None
        self.n_env = n_env

    def compute_action(self, obs, id, deterministic=True):
        assert len(obs) == len(id), f"{len(obs)} != len{id}"

        # compute action
        if self.policy_type in ["zoo", "aprl", "psu"]:
            obs_tf = np.zeros([self.n_env] + list(obs.shape[1:]))
            obs_tf[id] = obs
            act, state = self.policy.predict(
                obs_tf, state=self.state, deterministic=deterministic
            )
            act = act[id]
        else:
            if self.state is not None:
                last_state = (self.state[0][:, id], self.state[1][:, id])
            else:
                last_state = None
            if self.policy_type == "atla":
                with torch.no_grad():
                    mean, _, state = self.policy(obs, last_state)
                    act = to_numpy(mean)
            else:
                with torch.no_grad():
                    result = self.policy(
                        Batch(obs=obs), state=last_state, deterministic=deterministic
                    )
                act = self.policy.map_action(to_numpy(result.act))
                state = result.get("state", None)

        # update state
        if state is not None:
            if self.state is None:
                self.state = state
            else:
                self.state[0][:, id], self.state[1][:, id] = state
        return act

    def normalize_obs(self, obs):
        obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        obs = np.clip(obs, -10, 10)
        return obs

    def reset_state(self, id):
        if self.state is not None:
            h, c = self.state
            for i in id:
                h[:, i].zero_()
                c[:, i].zero_()
            self.state = h, c


def load_cityu_agent(
    venv,
    index=None,
    path=None,
    method="base",
    r=asdict(RepConf()),
    m=asdict(ActorCriticConf()),
    **kwargs,
):
    ob_space, ac_space = get_env_space(venv, index)
    encoder, actor, critic = get_model(ob_space, ac_space, **r, **m)
    kwargs_policy = dict(
        encoder=encoder,
        actor=actor,
        critic=critic,
        optim=None,
        dist_fn=get_dist_fn(ac_space),
        observation_space=ob_space,
        action_space=ac_space,
        **kwargs,
    )
    policy = POLICY_DICT[method](**kwargs_policy)
    ckpt = torch.load(path, map_location=m["device"])
    state_dict = policy.state_dict()
    state_dict.update({k: v for k, v in ckpt["model"].items() if k in state_dict})
    policy.load_state_dict(state_dict)
    return policy, ckpt["obs_rms"]


def load_aprl_agent(path, **kwargs):
    """Backwards compatibility hack to load old pickled policies
    which still expect modelfree.* to exist.
    """
    import aprl.training.scheduling  # noqa: F401

    # policy
    mock_modules = {
        "modelfree": "aprl",
        "modelfree.scheduling": "aprl.training.scheduling",
        "modelfree.training.scheduling": "aprl.training.scheduling",
    }
    for old, new in mock_modules.items():
        sys.modules[old] = sys.modules[new]
    model_path = os.path.join(path, "model.pkl")
    policy = stable_baselines.PPO2.load(model_path)
    for old in mock_modules:
        del sys.modules[old]

    # obs_rms
    try:
        normalize_path = os.path.join(path, "vec_normalize.pkl")
        with open(normalize_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        obs_rms = vec_normalize.obs_rms
    except FileNotFoundError:
        with open("{}/{}.pkl".format(path, "obs_rms"), "rb") as file_handler:
            obs_rms = pickle.load(file_handler)
            print("load obs_rms.pkl from", "{}/{}.pkl".format(path, "obs_rms"))
    return policy, obs_rms


def load_from_file(param_pkl_path):
    if param_pkl_path.endswith(".pkl"):
        with open(param_pkl_path, "rb") as f:
            params = pickle.load(f)
    else:
        params = np.load(param_pkl_path)
    return params


def load_from_model(param_pkl_path):
    if param_pkl_path.endswith(".pkl"):
        with open(param_pkl_path, "rb") as f:
            params = pickle.load(f)
        policy_param = params[1][0]
        flat_param = []
        for param in policy_param:
            flat_param.append(param.reshape(-1))
        flat_param = np.concatenate(flat_param, axis=0)
    else:
        flat_param = np.load(param_pkl_path, allow_pickle=True)
        if len(flat_param) == 3:
            flat_param_1 = []
            for i in flat_param[0]:
                flat_param_1.append(i)
            flat_param = []
            for param in flat_param_1:
                flat_param.append(param.reshape(-1))
            flat_param = np.concatenate(flat_param, axis=0)
    return flat_param


def load_zoo_agent(
    venv,
    index=0,
    tag="1",
    phase="train",
    param_path=None,
    obs_rms_path=None,
    **kwargs,
):
    """Loads a gym_compete zoo agent.
    :param venv: (gym.Env) the environment
    :param index: (int) the player ID of the agent we want to load ('0' or '1')
    :param tag: (str) version of the zoo agent (e.g. '1', '2', '3').
    :param phase: (str) variable scope for tensorflow
    :return a BaseModel, where predict executes the loaded policy."""
    env_name = venv.spec[0].id

    sess = make_session()
    with sess.as_default():
        # Build policy
        scope = f"zoo_policy_{tag}_{index}_{phase}"
        ob_space, ac_space = get_env_space(venv, index)
        kwargs = dict(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            n_env=len(venv),
            n_steps=1,
            n_batch=len(venv),
            scope=scope,
            reuse=tf.AUTO_REUSE,
        )
        policy_cls, policy_kwargs = get_policy_type_for_zoo_agent(env_name)
        kwargs.update(policy_kwargs)
        policy = policy_cls(**kwargs)

        # Load param and do modification if victim is retrained
        if param_path is None:
            params = load_zoo_agent_params(tag, env_name, index)
        else:
            print(f"load agent_{index+1} from {param_path}")
            params = load_from_model(param_path)
            canonical_env = env_name_to_canonical(env_name)
            variables_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)
            if canonical_env in POLICY_STATEFUL:
                if POLICY_STATEFUL[canonical_env]:
                    none_trainable_list = variables_list[:12]
                else:
                    none_trainable_list = variables_list[:6]
            else:
                msg = f"Unsupported Environment: {canonical_env}, choose from {POLICY_STATEFUL.keys()}"
                raise ValueError(msg)
            shapes = list(map(lambda x: x.get_shape().as_list(), none_trainable_list))
            untrainable_size = np.sum([int(np.prod(shape)) for shape in shapes])
            untrainable_param = load_from_file(obs_rms_path)[:untrainable_size]
            params = np.concatenate([untrainable_param, params])

        # Now restore params
        policy.restore(params)

        return base.PolicyToModel(policy)


def psu_set_from_flat(var_list, flat_params, sess=None):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    if total_size != flat_params.shape[0]:
        redundant = flat_params.shape[0] - total_size
        flat_params = flat_params[redundant:]
        assert flat_params.shape[0] == total_size, print(
            "Number of variables does not match when loading pretrained victim agents."
        )
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for shape, v in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start : start + size], shape)))
        start += size
    op = tf.group(*assigns)
    if sess is None:
        tf.get_default_session().run(op, {theta: flat_params})
    else:
        sess.run(op, {theta: flat_params})


def load_psu_agent(venv, index, path, **kwargs):
    try:
        param_path = os.path.join(path, "model.pkl")
    except:  # noqa: E722
        param_path = os.path.join(path, "model.npy")
    ob_rms_path = os.path.join(path, "obs_rms.pkl")
    sess = make_session()
    with sess.as_default():
        ob_space, ac_space = get_env_space(venv, index)
        kwargs = dict(
            sess=sess,
            ob_space=ob_space,
            ac_space=ac_space,
            n_env=len(venv),
            n_steps=1,
            n_batch=len(venv),
            reuse=tf.AUTO_REUSE,
        )
        policy = MlpPolicy(**kwargs)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")
        print(f"load agent_{index+1} from {param_path}")
        params = load_from_model(param_path)
        psu_set_from_flat(var_list, params)
        obs_rms = load_from_file(ob_rms_path)
        return base.PolicyToModel(policy), obs_rms


def load_atla_agent(venv, index, path, **kwargs):
    ob_space, ac_space = get_env_space(venv, index)
    obs_dim, act_dim = ob_space.shape[0], ac_space.shape[0]
    ckpt = torch.load(path, "cuda")
    policy = (
        CtsLSTMPolicy(obs_dim, act_dim)
        if "embedding_layer.weight" in ckpt["model"]
        else CtsPolicy(obs_dim, act_dim)
    ).to("cuda")
    policy.load_state_dict(ckpt["model"])
    return policy, ckpt["obs_rms"]


def load_agent(venv, index=None, policy_type="cityu", **kwargs):
    if policy_type == "zoo":
        policy = load_zoo_agent(venv, index, **kwargs)
        obs_rms = None
    elif policy_type == "aprl":
        policy, obs_rms = load_aprl_agent(**kwargs)
    elif policy_type == "cityu":
        policy, obs_rms = load_cityu_agent(venv, index, **kwargs)
    elif policy_type == "psu":
        policy, obs_rms = load_psu_agent(venv, index, **kwargs)
    elif policy_type == "atla":
        policy, obs_rms = load_atla_agent(venv, index, **kwargs)
    else:
        raise Exception("No support for such type of agent")
    ob_space = venv.observation_space
    n_env = len(ob_space) if isinstance(ob_space, list) else 1
    return Agent(policy_type, policy, obs_rms, n_env)
