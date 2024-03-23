import json
import os
from collections import defaultdict

from ap import WORKPLACE
from ap.envs.wrappers.sparse_reward import SPARSE_TASK
from ap.util.utils import find_all_files

SHORT_NAME_TO_GYM = {
    "hopper": "Hopper-v2",
    "walker": "Walker2d-v2",
    "half_cheetah": "HalfCheetah-v2",
    "halfcheetah": "HalfCheetah-v2",
    "ant": "Ant-v2",
}


def create_zoo(
    agent_type,
    root_dir=f"{WORKPLACE}/log/multirun",
    env_list=list(SPARSE_TASK.keys()),
    pattern=r".*?/(.*?)/(.*?)/(.*?)/seed=(\d+)",
    short_name=False,
):
    zoo = defaultdict(dict)
    for env_name in env_list:
        file_list, pattern_List = find_all_files(
            f"{root_dir}/{agent_type}",
            f"{env_name}/{pattern}",
            suffix="policy_latest.pth",
            return_pattern=True,
        )
        if short_name:
            name_list = [method for method, _, _, _ in pattern_List]
        else:
            name_list = [
                f"{method}-{tp}-{config}-seed{seed}".replace(",", "-").replace("=", "_")
                for method, tp, config, seed in pattern_List
            ]
        agent_list = {k: v for k, v in zip(name_list, file_list)}
        name_list.sort()
        for name in name_list:
            info = {
                "policy_type": "cityu",
                "path": agent_list[name],
                "method": name.split("_")[0],
            }
            zoo[f"{agent_type}/{env_name}"].update({name: info})

    path = f"{WORKPLACE}/zoo/{agent_type}"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/agents.json", "w") as f:
        json.dump(zoo, f, indent=4)


def create_robust_zoo(
    agent_type, root_dir=f"{WORKPLACE}/zoo", pattern=r"models/(.*?)/(.*).pth"
):
    path = f"{root_dir}/{agent_type}"
    zoo = defaultdict(dict)
    file_list, pattern_List = find_all_files(path, pattern, return_pattern=True)
    agent_list = {k: v for k, v in zip(pattern_List, file_list)}
    pattern_List.sort()
    for k in pattern_List:
        info = {"policy_type": "atla", "path": agent_list[k], "method": k[0]}
        zoo[f"{agent_type}/{SHORT_NAME_TO_GYM[k[1]]}"].update({k[0]: info})

    os.makedirs(path, exist_ok=True)
    with open(f"{path}/agents.json", "w") as f:
        json.dump(zoo, f, indent=4)


if __name__ == "__main__":
    create_zoo(
        "mujoco_sparse",
        env_list=list(SPARSE_TASK.keys()),
        short_name=True,
    )
    create_zoo(
        "mujoco",
        root_dir=f"{WORKPLACE}/zoo",
        env_list=["AntUMaze-v1", "Ant4Rooms-v1", "FetchReach-v1", "FetchPush-v1"],
        short_name=True,
    )
    create_robust_zoo("mujoco_robust")
