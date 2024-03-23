import gym
import mujoco_maze  # noqa

# "UMaze": [DistRewardUMaze, GoalRewardUMaze]
# env = gym.make("Ant4Rooms-v1")
env = gym.make("AntUMaze-v1")

# "4Rooms": [DistReward4Rooms, GoalReward4Rooms, SubGoal4Rooms]
# env = gym.make("Ant4Rooms-v1")

print(env.action_space)
print(env.observation_space)
print("goal", env.unwrapped._task.goals[0].pos)
print("init_pos", env.reset()[:2])

env.close()
