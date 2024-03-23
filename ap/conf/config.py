import re
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, List, Optional

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from ap import WORKPLACE
from ap.conf.collector_conf import *  # noqa: F403
from ap.conf.ppo_conf import *  # noqa: F403
from ap.util.utils import MLP_NORM

defaults = [
    {"c": "default"},
    {"p": "default"},
    {"override hydra/job_logging": "colorlog"},
    {"override hydra/hydra_logging": "colorlog"},
    "_self_",
]


@dataclass
class RepConf:
    use_rep: bool = False
    mlp_hidden_dim: int = 128
    obs_rep_dim: int = 64
    mlp_norm: str = "BN"

    def __post_init__(self):
        assert self.mlp_norm in MLP_NORM, "unsupported mlp_norm"


@dataclass
class ActorCriticConf:
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    device: str = "cuda"


@dataclass
class Config:
    c: CollectorConf = MISSING  # noqa: F405
    p: BasePPOConf = MISSING  # noqa: F405

    # model
    m: ActorCriticConf = field(default=ActorCriticConf)
    init_logsigma: float = float(np.log(1))
    lr: float = 3e-4
    last_a: bool = True  # last_layer_init_actor

    # logger and path
    comment: str = ""
    log_dir: str = "multirun"
    config_file: str = "config.json"
    resume_path: Optional[str] = None
    tag: Optional[str] = None

    # task
    task_type: str = "mujoco"
    task: Optional[str] = None
    seed: int = 0
    method: str = "base"
    vn: Optional[str] = None  # victim name
    tat: Optional[str] = None  # target agent type
    epsilon: Optional[float] = None
    is_bb: Optional[bool] = None  # is black box or not
    vx_ratio: float = 0.1

    # others
    show_progress: bool = False
    pid_bias: int = 0
    bind_core: bool = False
    save_state_freq: bool = False  # for heatmap
    state_map: bool = False
    test_after_train: bool = False
    eval: bool = False
    verbose: bool = False
    tl: Optional[int] = None  # time_limit

    # representation learning
    r: RepConf = field(default_factory=RepConf)

    # hydra setting
    hydra: DictConfig = OmegaConf.load(f"{WORKPLACE}/ap/conf/hydra.yaml")
    defaults: List[Any] = field(default_factory=lambda: defaults)

    def __post_init__(self):
        if self.c.name in ["eval", "video"]:
            self.eval = True
            self.config_file = f"config_{self.c.name}.json"
        if self.method in ["zero", "random"]:
            self.eval = True
        if self.resume_path is not None:
            epsilon = re.search(r"epsilon=(0.\d+)", self.resume_path)
            if epsilon is not None and self.epsilon is None:
                self.epsilon = float(epsilon.group(1))
            tat = re.search(r"tat=(.*?)(,|/)", self.resume_path)
            if tat is not None and self.tat is None:
                self.tat = re.sub(r"(,|/)", "", tat.group(1))
            vn = re.search(r"vn=(.*?)(,|/)", self.resume_path)
            if vn is not None and self.vn is None:
                self.vn = re.sub(r"(,|/)", "", vn.group(1))
                print(f"vn={vn}")
        if self.tag is not None:
            epsilon = re.search(r"epsilon_(0.\d+)", self.tag)
            if epsilon is not None and self.epsilon is None:
                self.epsilon = float(epsilon.group(1))
            tat = re.search(r"tat_(.*?)-", self.tag + "-")
            if tat is not None and self.tat is None:
                self.tat = tat.group(1).replace("/,", "")
            vn = re.search(r"vn_(.*)-", self.tag + "-")
            if vn is not None and self.vn is None:
                self.vn = vn.group(1)
        if self.resume_path or self.tag:
            print(f"epsilon={self.epsilon},tat={self.tat},vn={self.vn}")
        self.time_limit = self.tl


def register_configs():
    cs = ConfigStore.instance()
    cs.store(group="c", name="default", node=CollectorConf)
    cs.store(group="c", name="debug", node=DebugCollectorConf)
    cs.store(group="c", name="eval", node=EvalCollectorConf)
    cs.store(group="c", name="video", node=VideoCollectorConf)
    cs.store(group="c", name="aprl", node=APRLCollectorConf)
    cs.store(group="p", name="default", node=BasePPOConf)
    cs.store(name="config", node=Config)


def hydra_decorator(func: Callable) -> Callable:
    @wraps(func)
    def inner_decorator(cfg: DictConfig):
        cfg = OmegaConf.to_object(cfg)
        return func(cfg)

    return inner_decorator


if __name__ == "__main__":
    register_configs()

    @hydra.main(config_path=None, config_name="config")
    @hydra_decorator
    def my_conf(cfg):
        print(cfg)

    my_conf()
