from dataclasses import dataclass

import numpy as np


@dataclass
class CollectorConf:
    t: str = "5a6"
    max_epoch: int = 20
    spc: str = "128 * 64"
    nmb: int = 4  # nminibatches
    n_env: int = 64
    n_test_env: int = 64
    ept: int = 1000  # episode_per_test
    repeat_per_collect: int = 10
    video: bool = False
    name: str = "default"

    def __post_init__(self):
        self.total_timesteps = int(float(self.t.replace("a", "e")))
        self.nminibatches = self.nmb
        self.episode_per_test = self.ept
        self.step_per_collect = eval(self.spc.replace("a", "e"))
        self.step_per_epoch = self.total_timesteps // self.max_epoch
        self.buffer_size = self.step_per_collect
        self.minibatch_size = self.step_per_collect // self.nminibatches
        self.total_updates = (
            np.ceil(self.step_per_epoch / self.step_per_collect) * self.max_epoch
        )  # ~600


@dataclass
class DebugCollectorConf(CollectorConf):
    t: str = "3a3"
    max_epoch: int = 5
    spc: str = "1024"
    n_env: int = 3
    n_test_env: int = 3
    ept: int = 3
    name: str = "debug"


@dataclass
class EvalCollectorConf(CollectorConf):
    n_env: int = 1
    # n_test_env: int = 1
    name: str = "eval"


@dataclass
class VideoCollectorConf(CollectorConf):
    n_env: int = 1
    n_test_env: int = 1
    ept: int = 3
    video: bool = True
    name: str = "video"


APRL_CONF = dict(
    total_timesteps=int(20e6),
    batch_size=1024 * 16,  # 16384
    learning_rate=3e-4,
    nminibatches=4,
    noptepochs=4,
    num_env=8,
)


@dataclass
class APRLCollectorConf(CollectorConf):
    t: str = "2a7"
    spc: int = APRL_CONF["batch_size"]
    nmb: int = APRL_CONF["nminibatches"]
    n_env: int = APRL_CONF["num_env"]
    repeat_per_collect: int = APRL_CONF["noptepochs"]
    name: str = "APRL"
