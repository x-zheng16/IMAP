import copy

import numpy as np
from gym import Wrapper


class AnnotateState(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.v_before = [0, 0, 0]
        self.s0 = None
        try:
            self.goals = self.env._task.goals
        except:  # noqa: E722
            pass

    def step(self, act):
        try:
            if not hasattr(self, "wrapped_env"):
                mujoco_env = self.env
            else:
                mujoco_env = self.wrapped_env
            p_before = mujoco_env.get_body_com("torso").copy()
            obs, rew, done, info = super().step(act)
            p = mujoco_env.get_body_com("torso").copy()
            v = (p - p_before) / mujoco_env.dt
            a = (v - self.v_before) / mujoco_env.dt
            info.update(
                {
                    "rew_dense": rew,
                    "rew_risk": -np.linalg.norm(obs - self.s0),
                    "rew_death": -1,
                    "state": {
                        "x": p[0],
                        "y": p[1],
                        "z": p[2],
                        "vx": v[0],
                        "vy": v[1],
                        "vz": v[2],
                        "ax": a[0],
                        "ay": a[1],
                        "az": a[2],
                        "x_before": p_before[0],
                        "y_before": p_before[1],
                        "z_before": p_before[2],
                        "vx_before": self.v_before[0],
                        "vy_before": self.v_before[1],
                        "vz_before": self.v_before[2],
                    },
                }
            )
            self.v_before = v.copy()
        except:  # noqa: E722
            obs, rew, done, info = super().step(act)
            info.update(
                {
                    "rew_dense": rew,
                    "rew_risk": -np.linalg.norm(obs - self.s0),
                    "rew_death": -1,
                }
            )
            if self.spec.id.startswith("Fetch"):
                info.update(
                    {
                        "state": {
                            "x": obs[-6],
                            "y": obs[-5],
                            "z": obs[-4],
                            "dgx": obs[-3],
                            "dgy": obs[-2],
                            "dgz": obs[-1],
                        }
                    }
                )
            elif self.spec.id.startswith("Reacher"):
                info.update({"state": {"x": obs[-3], "y": obs[-2], "z": obs[-1]}})
        if "is_success" in info:
            info["success"] = info["is_success"]
        return obs, rew, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.s0 = copy.deepcopy(obs)
        return obs
