import os

import gym
import mujoco_py_131
import numpy as np
import pykeops
import tensorflow as tf
import torch

print("torch cuda: ", torch.cuda.is_available())
print("tf cuda: ", tf.test.is_gpu_available())

pykeops.test_numpy_bindings()
pykeops.test_torch_bindings()

env = gym.make("Ant-v3")
env.reset()
print("mujoco_py: obs=", env.step(env.action_space.sample())[0])
env.close()

humanoid_xml = f"{os.environ.get('MUJOCO_PY_131_MJPRO_PATH')}/model/humanoid.xml"
model = mujoco_py_131.MjModel(humanoid_xml)
obs = np.concatenate([model.data.qpos.flat, model.data.qvel.flat])
print("mujoco_py_131: obs = ", obs)
