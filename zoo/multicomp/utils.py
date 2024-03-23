import os

import gym_compete
from aprl.envs import VICTIM_INDEX
from aprl.envs.gym_compete import env_name_to_canonical, is_symmetric

from ap.envs.gym_compete.env_gym_compete import ENV_LIST

ENV_LIST_FOR_SHORT = {
    ENV_LIST[0]: ["kick", "KickAndDefend"],
    ENV_LIST[1]: ["humans", "SumoHumans"],
    ENV_LIST[2]: ["ants", "SumoAnts"],
    ENV_LIST[3]: ["you", "YouShallNotPass"],
}

SHORT_ENV_TO_FULL = {
    "KickAndDefend": ENV_LIST[0],
    "SumoHumans": ENV_LIST[1],
    "SumoAnts": ENV_LIST[2],
    "YouShallNotPass": ENV_LIST[3],
}

DEFAULT_TAG_INFO = {"victim": [1, 3, 1, 1], "adversary": [2, 1, 2, 1]}


def get_agent_path(env_name, tag, index=None):
    if index is None:
        index = VICTIM_INDEX[env_name]
    fname = f"agent{'' if is_symmetric(env_name) else index + 1}_parameters-v{tag}.pkl"
    return os.path.join(
        gym_compete.__path__[0], "agent_zoo", env_name_to_canonical(env_name), fname
    )
