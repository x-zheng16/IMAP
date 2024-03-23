from ap.envs.env_gym import make_venv_gym
from ap.envs.env_mujoco import make_venv_mujoco, make_venv_mujoco_eval
from ap.envs.env_mujoco_attack import make_venv_mujoco_attack
from ap.envs.venvs import DummyVectorEnv, ShmemVectorEnv
from ap.envs.wrappers import VenvNormObs, VenvVideoRecorder

ENV_F = {
    "gym": make_venv_gym,
    "mujoco": make_venv_mujoco,
    "mujoco_sparse": make_venv_mujoco,
    "mujoco_eval": make_venv_mujoco_eval,
    "mujoco_obs_attack": make_venv_mujoco_attack,
    "mujoco_act_attack": make_venv_mujoco_attack,
    "mujoco_sparse_obs_attack": make_venv_mujoco_attack,
    "mujoco_sparse_act_attack": make_venv_mujoco_attack,
}


def make_venv(cfg):
    kwargs_venv = dict(
        task_type=cfg.task_type,
        is_black_box=cfg.is_bb,
        target_agent_type=cfg.tat,
        env_name=cfg.task,
        n_env=cfg.c.n_env,
        n_test_env=cfg.c.n_test_env,
        pid_bias=cfg.pid_bias,
        bind_core=cfg.bind_core,
        victim_name=cfg.vn,
        epsilon=cfg.epsilon,
        vx_ratio=cfg.vx_ratio,
        venv_cls=DummyVectorEnv if cfg.c.video else ShmemVectorEnv,
        time_limit=cfg.time_limit,
    )
    venv, test_venv = ENV_F[cfg.task_type](**kwargs_venv)

    venv.seed(cfg.seed)
    test_venv.seed(cfg.seed)
    if cfg.c.video:
        venv = VenvVideoRecorder(venv)
        test_venv = VenvVideoRecorder(test_venv)
    if cfg.p.norm_obs:
        venv = VenvNormObs(venv)
        test_venv = VenvNormObs(test_venv, update_obs_rms=False)
        test_venv.set_obs_rms(venv.get_obs_rms())
    return venv, test_venv
