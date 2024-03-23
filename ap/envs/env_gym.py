import gym

from ap.envs.venvs import ShmemVectorEnv


def make_venv_gym(
    env_name,
    n_env=1,
    n_test_env=1,
    pid_bias=0,
    bind_core=False,
    venv_cls=ShmemVectorEnv,
    **kwargs,
):
    env_f = lambda: gym.make(env_name)  # noqa: E731
    venv = venv_cls([env_f for _ in range(n_env)], pid_bias, bind_core)
    test_venv = venv_cls([env_f for _ in range(n_test_env)], pid_bias, bind_core)
    return venv, test_venv
