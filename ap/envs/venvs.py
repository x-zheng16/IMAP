import os
from multiprocessing import Pipe
from multiprocessing.context import Process

import tianshou
from tianshou.env import worker
from tianshou.env.utils import CloudpickleWrapper
from tianshou.env.worker.subproc import _setup_buf, _worker


class BaseVectorEnv(tianshou.env.BaseVectorEnv):
    def __init__(self, env_fns, worker_fn, wait_num=None, timeout=None, **kwargs):
        self._env_fns = env_fns
        self.workers = [worker_fn(index, fn) for index, fn in enumerate(env_fns)]
        self.worker_class = type(self.workers[0])
        self.env_num = len(env_fns)
        self.wait_num = wait_num or len(env_fns)
        self.timeout = timeout
        self.is_async = self.wait_num != len(env_fns) or timeout is not None
        self.waiting_conn = []
        self.waiting_id = []
        self.ready_id = list(range(self.env_num))
        self.is_closed = False

    @property
    def unwrapped(self):
        return self


class SubprocEnvWorker(worker.SubprocEnvWorker):
    def __init__(
        self, env_fn, share_memory=False, bind_core=False, index=None, pid_bias=0
    ):
        self.parent_remote, self.child_remote = Pipe()
        self.share_memory = share_memory
        self.bind_core = bind_core
        self.buffer = None

        dummy = env_fn()
        obs_space = dummy.observation_space
        act_space = dummy.action_space
        dummy.close()
        del dummy

        self.buffer = _setup_buf(obs_space) if self.share_memory else None
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = Process(target=_worker, args=args, daemon=True)
        self.process.start()
        if self.bind_core:
            os.system(
                "taskset -p -c {:d} {} > /dev/null".format(
                    index % os.cpu_count() + pid_bias, self.process.pid
                )
            )
        self.child_remote.close()

        self._env_fn = env_fn
        self.is_closed = False
        self.observation_space = obs_space
        self.action_space = act_space
        self.is_reset = False


class DummyEnvWorker(worker.DummyEnvWorker):
    def __init__(self, env_fn, **kwargs):
        super().__init__(env_fn)


class DummyVectorEnv(BaseVectorEnv):
    def __init__(self, env_fns, pid_bias=0, bind_core=False, **kwargs):
        super().__init__(env_fns, lambda index, env_f: DummyEnvWorker(env_f), **kwargs)


class SubprocVectorEnv(BaseVectorEnv):
    def __init__(self, env_fns, pid_bias=0, bind_core=False, **kwargs):
        worker_fn = lambda index, env_f: SubprocEnvWorker(  # noqa: E731
            env_f, bind_core=bind_core, index=index, pid_bias=pid_bias
        )
        super().__init__(env_fns, worker_fn, **kwargs)


class ShmemVectorEnv(BaseVectorEnv):
    def __init__(self, env_fns, pid_bias=0, bind_core=False, **kwargs):
        worker_fn = lambda index, env_f: SubprocEnvWorker(  # noqa: E731
            env_f, True, bind_core=bind_core, index=index, pid_bias=pid_bias
        )
        super().__init__(env_fns, worker_fn, **kwargs)
