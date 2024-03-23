# IMAP

This is code for the DSN 2024 paper: "Toward Evaluating Robustness of Reinforcement Learning with Adversarial Policy". Currently, we only provide the code for IMAP against the RL agent in the single-agent RL tasks . We will release the code for IMAP against victim RL agent in the multi-agent competitive games soon.

## Environments

Though it's known to be challenging to install mujoco_py and mujoco_py_131, it should be successful if you strictly adhere to the following steps step by step. Thanks to conda, we can install both pytorch and tensorflow in the same environment.

```bash
# create a new conda env
conda create -n imap python=3.7
conda activate imap

# install torch and pykeops
pip install -U pip setuptools
pip install tianshou==0.4.10
pip install "hydra-core>=1.1,<1.2" "numpy>=1.16,<1.19" hydra-colorlog fast-histogram pykeops==2.1.2 pygame gym==0.15.4 seaborn sacred
conda install -c conda-forge libstdcxx-ng
python test/test_pykeops.py

# install gym_compete first
pip install gym_compete@git+https://github.com/HumanCompatibleAI/multiagent-competition.git@3a3f9dc
pip uninstall tensorflow # stable-baselines use tensorflow==1.15.5 instead of tensorflow==2.11.0

# install mujoco_py and mujoco_py_131 for single-agent environments
sudo apt install libglew-dev libosmesa6-dev patchelf libgl1-mesa-glx libgl1-mesa-dev libglfw3 libglu1-mesa libxrandr2 libxinerama1 libxi6 libxcursor-dev xvfb ffmpeg

# Please refer to README.md in mujoco_py_131 for how to set the environment variables.
pip install -e dependency/mujoco_py_131 
pip install cffi "Cython<3" glfw imageio
pip install "mujoco-py<2,>=1.15.1.68"
pip install mujoco_maze

# install tensorflow and sb
pip install tensorflow-gpu==1.15.5 tensorflow==1.15.5
pip install git+ssh://git@github.com/hill-a/stable-baselines.git@v2.10.1
conda install cudatoolkit=10.0 cudnn~=7.6
python test/test_env.py # You should successfully run mujoco_py, mujoco_py_131, tensorflow, torch, and pykeops!!!

# install multi-agent competition
# please refer to dependency/adversarial-policies/README.md for how to install the multi-agent competition
pip install -e dependency/adversarial-policies

# install the imap
pip install -e .

# create agents.json for mujoco_robust and mujoco_sparse
python zoo/create_zoo.py

# create victim_zoo
cd zoo/multicomp
python victim_zoo.py

# test vanilla PPO method now!
python ap/train.py -m log_dir=debug c=debug task=Hopper-v3 method=base

```

## Experiments

### launch IMAP to attack a black-box robust dense-reward RL agent

```bash
# SA-RL
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=PPO method=base seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=ATLA-PPO method=base seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=SAPPO method=base seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=RADIAL method=base seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=WocaR-PPO method=base seed=0,100,200

# IMAP-SC
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=PPO method=imap p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=ATLA-PPO method=imap p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=SAPPO method=imap p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=RADIAL method=imap p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=WocaR-PPO method=imap p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=ATLA-LSTM-SAPPO method=imap p.s=PConst seed=0,100,200

# IMAP-PC 
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=PPO method=imap p.s=PConst seed=0,100,200 p.cs=3a6 # we search the optimal p.cs from [1a4,3a4,1a5,3a5,1a6,3a6]
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=ATLA-PPO method=imap p.s=PConst seed=0,100,200 p.cs=3a6
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=SAPPO method=imap p.s=PConst seed=0,100,200 p.cs=3a6
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=RADIAL method=imap p.s=PConst seed=0,100,200 p.cs=3a6
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=WocaR-PPO method=imap p.s=PConst seed=0,100,200 p.cs=3a6

# IMAP-R
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=PPO method=dense p.drt=risk p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=ATLA-PPO method=dense p.drt=risk p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=SAPPO method=dense p.drt=risk p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=RADIAL method=dense p.drt=risk p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=WocaR-PPO method=dense p.drt=risk p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=ATLA-LSTM-SAPPO method=dense p.drt=risk p.s=PConst seed=0,100,200

# IMAP-D
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=PPO method=div p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=ATLA-PPO method=div p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=SAPPO method=div p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=RADIAL method=div p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=WocaR-PPO method=div p.s=PConst seed=0,100,200
python ap/train.py -m task_type=mujoco_obs_attack task=HalfCheetah-v2 epsilon=0.15 tat=mujoco_robust vn=ATLA-LSTM-SAPPO method=div p.s=PConst seed=0,100,200
```
