# Intrinsically Motivated Adversarial Policy

**Toward Evaluating Robustness of Reinforcement Learning with Adversarial Policy (DSN 2024)** \[[Paper](https://arxiv.org/pdf/2305.02605)\]  
[Xiang Zheng](https://x-zheng16.github.io), [Xingjun Ma](http://xingjunma.com), [Shengjie Wang](https://shengjiewang-jason.github.io), Xinyu Wang, Chao Shen, [Cong Wang](https://www.cs.cityu.edu.hk/~congwang/)

## Abstract

Reinforcement learning agents are susceptible to evasion attacks during deployment. In single-agent environments, these attacks can occur through imperceptible perturbations injected into the inputs of the victim policy network. In multi-agent environments, an attacker can manipulate an adversarial opponent to influence the victim policy's observations indirectly. While adversarial policies offer a promising technique to craft such attacks, current methods are either sample-inefficient due to poor exploration strategies or require extra surrogate model training under the black-box assumption. To address these challenges, in this paper, we propose Intrinsically Motivated Adversarial Policy (IMAP) for efficient black-box adversarial policy learning in both single- and multi-agent environments. We formulate four types of adversarial intrinsic regularizers—maximizing the adversarial state coverage, policy coverage, risk, or divergence—to discover potential vulnerabilities of the victim policy in a principled way. We also present a novel bias-reduction method to balance the extrinsic objective and the adversarial intrinsic regularizers adaptively. Our experiments validate the effectiveness of the four types of adversarial intrinsic regularizers and the bias-reduction method in enhancing black-box adversarial policy learning across a variety of environments. Our IMAP successfully evades two types of defense methods, adversarial training and robust regularizer, decreasing the performance of the state-of-the-art robust WocaR-PPO agents by 34\%-54\% across four single-agent tasks. IMAP also achieves a state-of-the-art attacking success rate of 83.91\% in the multi-agent game YouShallNotPass.

## Environments

```bash
# Create conda env
conda create -n imap python=3.7
conda activate imap

# Install torch and pykeops
pip install -U pip setuptools
pip install tianshou==0.4.10
pip install "hydra-core>=1.1,<1.2" "numpy>=1.16,<1.19" hydra-colorlog fast-histogram pykeops==2.1.2 pygame gym==0.15.4 seaborn sacred
conda install -c conda-forge libstdcxx-ng
python test/test_pykeops.py

# Install gym_compete
pip install gym_compete@git+https://github.com/HumanCompatibleAI/multiagent-competition.git@3a3f9dc
pip uninstall tensorflow # stable-baselines use tensorflow==1.15.5 instead of tensorflow==2.11.0

# Install mujoco_py and mujoco_py_131 for single-agent environments
sudo apt install libglew-dev libosmesa6-dev patchelf libgl1-mesa-glx libgl1-mesa-dev libglfw3 libglu1-mesa libxrandr2 libxinerama1 libxi6 libxcursor-dev xvfb ffmpeg

# Please refer to README.md in mujoco_py_131 for how to set the environment variables
pip install -e dependency/mujoco_py_131 
pip install cffi "Cython<3" glfw imageio
pip install "mujoco-py<2,>=1.15.1.68"
pip install mujoco_maze

# Install tensorflow and sb
pip install tensorflow-gpu==1.15.5 tensorflow==1.15.5
pip install git+ssh://git@github.com/hill-a/stable-baselines.git@v2.10.1
conda install cudatoolkit=10.0 cudnn~=7.6
python test/test_env.py # You should successfully run mujoco_py, mujoco_py_131, tensorflow, torch, and pykeops!!!

# Install multi-agent competition
# please refer to dependency/adversarial-policies/README.md for how to install the multi-agent competition
pip install -e dependency/adversarial-policies

# Install IMAP
pip install -e .

# Create agents.json for mujoco_robust and mujoco_sparse
python zoo/create_zoo.py

# Create victim_zoo
cd zoo/multicomp
python victim_zoo.py

# Test vanilla PPO
python ap/train.py -m log_dir=debug c=debug task=Hopper-v3 method=base
```

## Experiments

### IMAP Against Robust MuJoCo RL Agents in Single-Agent Tasks

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
