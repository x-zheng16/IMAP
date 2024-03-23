from dataclasses import dataclass

IN_REW_SCHEDULE = ["EX", "RF", "PConst", "PLinear", "PExp", "L"]


@dataclass
class BasePPOConf:
    # basic
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    eps_clip: float = 0.2
    norm_obs: bool = True
    norm_return: bool = True
    norm_adv: bool = True
    rc_adv: bool = False
    action_bound_method: str = "clip"
    action_scaling: bool = True
    deterministic_eval: bool = True

    # pbe
    cs: str = "1a4"  # buffer size for entropy estimator
    sa_ent: bool = False
    k: int = 10  # k-nearest neighbour
    style: str = "log_mean"

    # dense
    drt: str = "original"

    # schedule for balancing in_rew and ex_rew
    s: str = "EX"  # ["EX"]
    rf_rate: float = 0  # determining proportion of reward-free pre-training stage (0,1]
    ex_rate: float = 1.2  # estimating upper_bound of last_ex_return [1,inf)
    lag_rate: float = 10  # update step of Lagrangian coefficient

    def __post_init__(self):
        self.cache_size = int(float(self.cs.replace("a", "e")))
        assert self.s in IN_REW_SCHEDULE, "unsupported in_rew schedule"
        assert self.drt is None or self.drt in [
            "original",
            "dist",
            "vel",
            "risk",
            "death",
        ], "unsupported dense reward type"
        self.dense_reward_type = self.drt
