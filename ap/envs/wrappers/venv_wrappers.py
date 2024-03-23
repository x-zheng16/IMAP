import os

from gym.wrappers.monitoring import video_recorder
from tianshou.env.venvs import GYM_RESERVED_KEYS, BaseVectorEnv
from tianshou.utils import RunningMeanStd

from ap.envs.venvs import DummyVectorEnv


class VenvWrapper(BaseVectorEnv):
    def __init__(self, venv: BaseVectorEnv):
        self.venv = venv
        self.is_async = venv.is_async

    def __len__(self):
        return len(self.venv)

    def __getattribute__(self, key):
        if key in GYM_RESERVED_KEYS:
            return getattr(self.venv, key)
        else:
            return super().__getattribute__(key)

    def get_env_attr(self, key, id=None):
        return self.venv.get_env_attr(key, id)

    def set_env_attr(self, key, value, id=None):
        return self.venv.set_env_attr(key, value, id)

    def reset(self, id=None, **kwargs):
        return self.venv.reset(id, **kwargs)

    def step(self, action, id=None):
        return self.venv.step(action, id)

    def seed(self, seed=None):
        return self.venv.seed(seed)

    def render(self, **kwargs):
        return self.venv.workers[0].render()

    def close(self):
        self.venv.close()

    @property
    def unwrapped(self):
        return self.venv.unwrapped


class VenvNormObs(VenvWrapper):
    def __init__(self, venv, update_obs_rms=True):
        super().__init__(venv)
        self.update_obs_rms = update_obs_rms
        self.obs_rms = RunningMeanStd()

    def step(self, act, id=None):
        results = super().step(act, id)
        return (self._norm_obs(results[0]), *results[1:])

    def reset(self, id=None, **kwargs):
        rval = super().reset(id, **kwargs)
        returns_info = isinstance(rval, (tuple, list)) and (len(rval) == 2)
        obs = rval[0] if returns_info else rval
        obs = self._norm_obs(obs)
        return (obs, rval[1]) if returns_info else obs

    def _norm_obs(self, obs):
        if self.update_obs_rms:
            self.obs_rms.update(obs)
        return self.obs_rms.norm(obs)

    def set_obs_rms(self, obs_rms):
        self.obs_rms = obs_rms

    def get_obs_rms(self):
        return self.obs_rms


class VenvVideoRecorder(VenvWrapper):
    def __init__(self, venv, directory="video", single_video=False):
        super().__init__(venv)
        self.episode_id = 0
        self.env_id = 0
        self.video_recorder = None
        self.single_video = single_video
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)

    def step(self, act, id=None):
        id = self.venv._wrap_id(id)
        obs, rew, done, info = super().step(act, id)
        if self.env_id in id:
            self.video_recorder.capture_frame()
        if done[self.env_id]:
            winner_id = info[self.env_id].get("winner_id", None)
            metadata = {f"winner in # {self.episode_id}": winner_id}
            self.video_recorder.metadata.update(metadata)
        return obs, rew, done, info

    def reset(self, id=None, **kwargs):
        self._reset_video_recorder()
        self.episode_id += 1
        return super().reset(id, **kwargs)

    def _reset_video_recorder(self):
        if self.video_recorder and not self.single_video:
            self.video_recorder.close()
            self.video_recorder = None
        if self.video_recorder is None:
            vn = "video" if self.single_video else "video.{:d}".format(self.episode_id)
            assert isinstance(self.unwrapped, DummyVectorEnv)
            self.video_recorder = video_recorder.VideoRecorder(
                env=self.unwrapped.workers[self.env_id].env,
                base_path=os.path.join(self.directory, vn),
            )
